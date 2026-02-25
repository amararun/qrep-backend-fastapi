"""
QREP Backend API - Portfolio Analytics (Powered by QuantStats)

This backend provides portfolio analysis endpoints powered by the original ranaroussi/quantstats library.
Separate from the main QUANTSTATS backend which uses quantstats-lumi.

Features:
- Rate limiting (configurable via RATE_LIMIT env var)
- API key protection (QPULSE_API_KEY env var)
- CORS wildcard origins (credentials=False, API-key auth only)
- Logging via tigzig-api-monitor (optional)
"""

from fastapi import FastAPI, HTTPException, Request, Header, Query
from fastapi_mcp import FastApiMCP
import httpx
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import asyncio
import quantstats as qs
import yfinance as yf
import pandas as pd
import numpy as np
import os
import logging
import warnings
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ReportLab for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from io import BytesIO
from bs4 import BeautifulSoup

# Optional: tigzig-api-monitor for centralized logging
try:
    from tigzig_api_monitor import APIMonitorMiddleware
    HAS_LOGGER = True
except ImportError:
    HAS_LOGGER = False

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# API Key for endpoint protection
QPULSE_API_KEY = os.getenv("QPULSE_API_KEY", "")

# Rate limiting
RATE_LIMIT = os.getenv("RATE_LIMIT", "20/minute")
GLOBAL_RATE_LIMIT = os.getenv("GLOBAL_RATE_LIMIT", "100/minute")

# Concurrency limits (compute-heavy endpoints)
MAX_CONCURRENT_PER_IP = int(os.getenv("MAX_CONCURRENT_PER_IP", "2"))
MAX_CONCURRENT_GLOBAL = int(os.getenv("MAX_CONCURRENT_GLOBAL", "10"))

# Base URL for report URLs (set in Coolify env vars)
BASE_URL = os.getenv("BASE_URL", "https://qpulse-api.tigzig.com")

# CORS: wildcard origins with credentials=False (API-key auth, no cookies needed)

# Symbol validation regex
SYMBOL_PATTERN = re.compile(r'^[A-Za-z0-9.\-=^$]{1,20}$')

# Cleanup API key (for cron-job.org to call /cleanup endpoint)
CLEANUP_API_KEY = os.getenv("CLEANUP_API_KEY", "")

# Directories
REPORTS_DIR = "static/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Monkey patch numpy.product if needed
if not hasattr(np, 'product'):
    np.product = np.prod

# =============================================================================
# CLIENT IP EXTRACTION (Cloudflare / proxy aware)
# =============================================================================

def get_real_client_ip(request: Request) -> str:
    """Extract real client IP, aware of our proxy chain.

    Network topology (normal):  Browser -> Cloudflare -> Caddy -> Container
    Network topology (no CF):   Browser -> Caddy -> Container

    Behind Caddy, request.client.host is always 172.18.0.10 (Docker proxy).

    With Cloudflare (orange cloud): CF-Connecting-IP = real user IP.
      Caddy XFF = Cloudflare edge IP (172.70.x.x), NOT the real user.
    Without Cloudflare (grey cloud): CF-Connecting-IP absent.
      Caddy XFF = real user IP (browser's actual IP is the TCP source Caddy sees).

    So XFF is needed as fallback for when Cloudflare is removed.
    """
    # 1. X-Tigzig-User-IP: Real IP forwarded by Vercel serverless (Pattern B)
    #    For multi-hop: Browser -> CF -> Vercel -> CF -> Hetzner
    #    CF-Connecting-IP gets overwritten to Vercel's IP at second CF pass.
    #    Vercel function captures real IP and forwards it in this custom header.
    tigzig_ip = request.headers.get("x-tigzig-user-ip", "").strip()
    if tigzig_ip:
        return tigzig_ip
    # 2. CF-Connecting-IP: Cloudflare sets this to the real client IP (Pattern A)
    #    For direct: Browser -> CF -> Hetzner. Unspoofable with orange cloud on.
    cf_ip = request.headers.get("cf-connecting-ip", "").strip()
    if cf_ip:
        return cf_ip
    # 3. X-Forwarded-For: Fallback for when Cloudflare is OFF (grey cloud).
    #    Without CF, Caddy sees browser's real IP as TCP source and sets XFF to it.
    #    With CF on, this would be a Cloudflare edge IP (172.70.x.x) - but we
    #    never reach here when CF is on because step 2 catches it.
    xff = request.headers.get("x-forwarded-for", "").strip()
    if xff:
        return xff.split(",")[0].strip()
    # 4. Last resort: TCP socket source (172.18.0.10 behind Caddy - useless for rate limiting)
    return request.client.host if request.client else "unknown"


# =============================================================================
# CONCURRENCY TRACKING
# =============================================================================

_active_queries: Dict[str, int] = {}
_active_global: int = 0
_concurrency_lock = asyncio.Lock()


async def check_concurrency(client_ip: str):
    """Check per-IP and global concurrency limits. Raises HTTPException if exceeded."""
    global _active_global
    async with _concurrency_lock:
        if _active_global >= MAX_CONCURRENT_GLOBAL:
            raise HTTPException(status_code=503, detail="Server is at capacity. Please try again in a few seconds.")
        ip_count = _active_queries.get(client_ip, 0)
        if ip_count >= MAX_CONCURRENT_PER_IP:
            raise HTTPException(status_code=429, detail="Too many concurrent requests. Please try again shortly.")
        _active_queries[client_ip] = ip_count + 1
        _active_global += 1


async def _release_concurrency_inner(client_ip: str):
    global _active_global
    async with _concurrency_lock:
        _active_queries[client_ip] = max(0, _active_queries.get(client_ip, 1) - 1)
        if _active_queries[client_ip] == 0:
            _active_queries.pop(client_ip, None)
        _active_global = max(0, _active_global - 1)


async def release_concurrency(client_ip: str):
    """Release concurrency slot. Uses asyncio.shield to survive client disconnects."""
    try:
        await asyncio.shield(_release_concurrency_inner(client_ip))
    except asyncio.CancelledError:
        pass

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sanitize_for_filename(text: str) -> str:
    """Sanitize text for use in filename"""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', text)


def cleanup_old_reports(max_age_days: int = 3):
    """Clean up report files older than max_age_days"""
    try:
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        cutoff_timestamp = cutoff_time.timestamp()

        if os.path.exists(REPORTS_DIR):
            deleted_count = 0
            for filename in os.listdir(REPORTS_DIR):
                file_path = os.path.join(REPORTS_DIR, filename)
                if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_timestamp:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting {filename}: {e}")

            if deleted_count > 0:
                logger.info(f"Cleanup: Deleted {deleted_count} old report files")
    except Exception as e:
        logger.error(f"Cleanup error: {e}")


def process_stock_data(data, symbol: str) -> tuple:
    """Process yfinance data to get returns series and close prices.
    Returns (returns_series, close_prices_series)."""
    try:
        # Get Close prices
        if isinstance(data.columns, pd.MultiIndex):
            prices = data[('Close', symbol)]
        else:
            prices = data['Close']

        # Calculate returns
        returns = prices.pct_change().dropna()

        # Convert to Series if DataFrame
        if isinstance(returns, pd.DataFrame):
            returns = returns.squeeze()
        if isinstance(prices, pd.DataFrame):
            prices = prices.squeeze()

        returns.name = symbol
        prices.name = symbol
        return returns, prices
    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        raise


def _clean_metric_value(value):
    """Clean up a metric value for JSON serialization"""
    if isinstance(value, str):
        value = value.replace('%', '').strip()
        if value == '-':
            return None
        else:
            try:
                return float(value)
            except:
                return value
    elif pd.isna(value):
        return None
    return value


def get_metrics_for_symbol(returns: pd.Series, benchmark_returns: pd.Series, rf: float = 0.0, omega_threshold: Optional[float] = None, var_confidence: Optional[float] = None, tail_cutoff: Optional[float] = None) -> Dict[str, Any]:
    """
    Get all metrics for a symbol using qs.reports.metrics()
    Returns dict with BOTH strategy metrics AND benchmark metrics from the same QuantStats call.
    This is the native QuantStats output - no custom calculations.

    Args:
        omega_threshold: Optional custom threshold for Omega ratio (as decimal, e.g., 0.05 for 5%)
        var_confidence: Optional confidence level for VaR/CVaR (as decimal, e.g., 0.99 for 99%)
        tail_cutoff: Optional percentile for Tail Ratio calculation (as decimal, e.g., 0.99 for 99%)
    """
    try:
        # Get full metrics as DataFrame from QuantStats
        # This returns a DataFrame with 2 columns: [Benchmark, Strategy]
        metrics_df = qs.reports.metrics(
            returns=returns,
            benchmark=benchmark_returns,
            rf=rf,
            display=False,
            mode='full',
            compounded=True,
            periods_per_year=252,
            prepare_returns=False
        )

        # metrics_df has index as metric names, columns: [Benchmark, Strategy]
        # Column 0 = Benchmark metrics, Column 1 = Strategy metrics
        has_benchmark = len(metrics_df.columns) > 1
        strategy_col = metrics_df.columns[1] if has_benchmark else metrics_df.columns[0]
        benchmark_col = metrics_df.columns[0] if has_benchmark else None

        # Extract strategy metrics
        strategy_metrics = {}
        for metric_name in metrics_df.index:
            clean_name = metric_name.strip()
            if clean_name:  # Skip empty separator rows
                value = metrics_df.loc[metric_name, strategy_col]
                strategy_metrics[clean_name] = _clean_metric_value(value)

        # Override affected metrics with direct qs.stats.* calls on the Series
        # These 6 metrics are affected by dropna() behavior when processing DataFrame
        # See docs/METRICS.md and projects/validations/METRICS_ANALYSIS.md for details
        try:
            # Avg. Return - shown as percentage in QuantStats output
            avg_return_val = qs.stats.avg_return(returns)
            if avg_return_val is not None and not pd.isna(avg_return_val):
                strategy_metrics["Avg. Return"] = round(avg_return_val * 100, 2)

            # Avg. Win - shown as percentage
            avg_win_val = qs.stats.avg_win(returns)
            if avg_win_val is not None and not pd.isna(avg_win_val):
                strategy_metrics["Avg. Win"] = round(avg_win_val * 100, 2)

            # Avg. Loss - shown as percentage (negative)
            avg_loss_val = qs.stats.avg_loss(returns)
            if avg_loss_val is not None and not pd.isna(avg_loss_val):
                strategy_metrics["Avg. Loss"] = round(avg_loss_val * 100, 2)

            # Payoff Ratio
            payoff_val = qs.stats.payoff_ratio(returns)
            if payoff_val is not None and not pd.isna(payoff_val):
                strategy_metrics["Payoff Ratio"] = round(payoff_val, 2)

            # CPC Index - uses profit_factor * win_rate * payoff_ratio internally
            cpc_val = qs.stats.cpc_index(returns)
            if cpc_val is not None and not pd.isna(cpc_val):
                strategy_metrics["CPC Index"] = round(cpc_val, 2)

            # Omega ratio with custom threshold (if provided)
            if omega_threshold is not None:
                omega_val = qs.stats.omega(returns, rf=omega_threshold)
                if omega_val is not None and not pd.isna(omega_val):
                    strategy_metrics["Omega"] = round(omega_val, 2)
                    logger.info(f"Omega computed with custom threshold {omega_threshold*100:.2f}%: {omega_val:.2f}")

            # VaR and CVaR with custom confidence level (if provided)
            if var_confidence is not None:
                # Daily Value-at-Risk - shown as percentage (negative)
                var_val = qs.stats.var(returns, confidence=var_confidence)
                if var_val is not None and not pd.isna(var_val):
                    strategy_metrics["Daily Value-at-Risk"] = round(var_val * 100, 2)
                    logger.info(f"VaR computed with confidence {var_confidence*100:.0f}%: {var_val*100:.2f}%")

                # Expected Shortfall (cVaR) - shown as percentage (negative)
                cvar_val = qs.stats.cvar(returns, confidence=var_confidence)
                if cvar_val is not None and not pd.isna(cvar_val):
                    strategy_metrics["Expected Shortfall (cVaR)"] = round(cvar_val * 100, 2)
                    logger.info(f"CVaR computed with confidence {var_confidence*100:.0f}%: {cvar_val*100:.2f}%")

            # Tail Ratio with custom cutoff (if provided)
            if tail_cutoff is not None:
                tail_val = qs.stats.tail_ratio(returns, cutoff=tail_cutoff)
                if tail_val is not None and not pd.isna(tail_val):
                    strategy_metrics["Tail Ratio"] = round(tail_val, 2)
                    logger.info(f"Tail Ratio computed with cutoff {tail_cutoff*100:.0f}%: {tail_val:.2f}")

        except Exception as e:
            logger.warning(f"Error computing override metrics: {e}")
            # Keep the original qs.reports.metrics() values if override fails

        # Extract benchmark metrics (from the SAME QuantStats call - native output)
        benchmark_metrics = {}
        if benchmark_col is not None:
            for metric_name in metrics_df.index:
                clean_name = metric_name.strip()
                if clean_name:
                    value = metrics_df.loc[metric_name, benchmark_col]
                    benchmark_metrics[clean_name] = _clean_metric_value(value)

        # Categorize metrics for tabs
        strategy_categorized = categorize_metrics(strategy_metrics)
        benchmark_categorized = categorize_metrics(benchmark_metrics) if benchmark_metrics else None

        return {
            'all_metrics': strategy_metrics,
            'categorized': strategy_categorized,
            'benchmark_all_metrics': benchmark_metrics,
            'benchmark_categorized': benchmark_categorized
        }

    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise


def categorize_metrics(metrics_dict: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Categorize metrics into groups for tabs UI
    """
    categories = {
        'overview': {
            'title': 'Overview',
            'patterns': ['Start Period', 'End Period', 'Risk-Free Rate', 'Time in Market',
                        'Cumulative Return', 'CAGR', 'Sharpe', 'Sortino', 'Max Drawdown',
                        'Volatility (ann.)']
        },
        'risk': {
            'title': 'Risk Metrics',
            'patterns': ['Volatility (ann.)', 'Skew', 'Kurtosis', 'Daily Value-at-Risk',
                        'Expected Shortfall', 'Ulcer Index', 'Serenity Index',
                        'Risk of Ruin', 'Tail Ratio', 'Kelly Criterion']
        },
        'ratios': {
            'title': 'Risk-Adjusted Ratios',
            'patterns': ['Sharpe', 'Prob. Sharpe', 'Smart Sharpe', 'Sortino',
                        'Smart Sortino', 'Sortino/√2', 'Omega', 'Calmar', 'Information Ratio',
                        'Treynor Ratio']
        },
        'drawdown': {
            'title': 'Drawdown Analysis',
            'patterns': ['Max Drawdown', 'Longest DD', 'Avg. Drawdown', 'Recovery Factor']
        },
        'returns': {
            'title': 'Period Returns',
            'patterns': ['MTD', '3M', '6M', 'YTD', '1Y', '3Y (ann.)', '5Y (ann.)',
                        '10Y (ann.)', 'All-time (ann.)', 'Best Day', 'Worst Day',
                        'Best Month', 'Worst Month', 'Best Year', 'Worst Year',
                        'Expected Daily', 'Expected Monthly', 'Expected Yearly']
        },
        'winloss': {
            'title': 'Win/Loss Analysis',
            'patterns': ['Win Days', 'Win Month', 'Win Quarter', 'Win Year',
                        'Avg. Up Month', 'Avg. Down Month', 'Max Consecutive',
                        'Payoff Ratio', 'Profit Factor', 'Gain/Pain', 'Common Sense',
                        'CPC Index', 'Outlier Win', 'Outlier Loss']
        },
        'benchmark': {
            'title': 'Benchmark Comparison',
            'patterns': ['Beta', 'Alpha', 'Correlation', 'R^2', 'Information Ratio',
                        'Treynor']
        }
    }

    categorized = {}
    for cat_key, cat_info in categories.items():
        cat_metrics = {}
        for metric_name, value in metrics_dict.items():
            for pattern in cat_info['patterns']:
                if pattern.lower() in metric_name.lower():
                    cat_metrics[metric_name] = value
                    break
        categorized[cat_key] = {
            'title': cat_info['title'],
            'metrics': cat_metrics
        }

    return categorized


def add_qrep_branding(html_file_path: str):
    """Add simple header to generated HTML report - raw QuantStats output"""
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Simple, clean header - black text, left-aligned
        header = '''
        <div style="
            padding: 16px 24px;
            margin-bottom: 16px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            border-bottom: 2px solid #000000;
        ">
            <div style="font-size: 17px; color: #000000;">
                Original QuantStats HTML tearsheet, As-Is &nbsp;&nbsp;|&nbsp;&nbsp; Generated on <a href="https://tigzig.com" target="_blank" style="color: #000000; text-decoration: underline;">TIGZIG</a> - AI for Analytics - Apps Suite
            </div>
        </div>
        '''

        # Insert header after <body>
        if '<body' in html_content:
            body_start = html_content.find('<body')
            body_tag_end = html_content.find('>', body_start) + 1
            html_content = html_content[:body_tag_end] + header + html_content[body_tag_end:]

        # No footer - keep the raw QuantStats output clean

        with open(html_file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    except Exception as e:
        logger.error(f"Error adding branding: {e}")


# =============================================================================
# RATE LIMIT HANDLER
# =============================================================================

def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning(f"Rate limit exceeded for IP: {get_real_client_ip(request)}")
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."}
    )


# =============================================================================
# LIFESPAN
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("QREP Backend API starting...")
    logger.info(f"Using original QuantStats library v{qs.__version__}")
    logger.info(f"Rate limit: {RATE_LIMIT}")
    if not QPULSE_API_KEY:
        logger.critical("QPULSE_API_KEY is not set! All authenticated endpoints will return 503.")
    logger.info(f"API key protection: {'ENABLED' if QPULSE_API_KEY else 'NOT SET - ENDPOINTS WILL REJECT REQUESTS'}")
    logger.info(f"Base URL: {BASE_URL}")

    # Cleanup old reports
    cleanup_old_reports()

    yield

    logger.info("QREP Backend API shutting down...")
    logger.info("=" * 50)


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="QREP Backend API",
    description="Portfolio analytics API powered by QuantStats",
    version="1.0.0",
    lifespan=lifespan
)

# Rate limiter
limiter = Limiter(key_func=get_real_client_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_handler)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: API Monitor middleware for logging
if HAS_LOGGER:
    app.add_middleware(
        APIMonitorMiddleware,
        app_name="QPULSE_ORIGINAL",
        include_prefixes=("/qpulse/", "/mcp/"),
    )

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# =============================================================================
# API KEY VALIDATION
# =============================================================================

def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Verify API key - always enforced, never optional"""
    if not QPULSE_API_KEY:
        raise HTTPException(status_code=503, detail="Service misconfigured")
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    if x_api_key != QPULSE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


def validate_symbol(symbol: str) -> str:
    """Validate and sanitize a stock symbol. Returns cleaned symbol or raises."""
    cleaned = symbol.strip().upper()
    if not SYMBOL_PATTERN.match(cleaned):
        raise HTTPException(status_code=400, detail=f"Invalid symbol format: {symbol[:20]}")
    return cleaned


# =============================================================================
# MODELS
# =============================================================================

class QREPAnalysisRequest(BaseModel):
    """Request model for QREP tearsheet generation"""
    symbol: str = Field(description="Stock symbol (e.g., 'AAPL', 'MSFT')")
    benchmark: str = Field(default="SPY", description="Benchmark symbol")
    start_date: str = Field(description="Start date (YYYY-MM-DD)")
    end_date: str = Field(description="End date (YYYY-MM-DD)")
    risk_free_rate: float = Field(default=0.0, description="Risk-free rate as decimal (e.g., 0.045 for 4.5%)")


class QREPAnalysisResponse(BaseModel):
    """Response model for QREP tearsheet"""
    success: bool
    html_url: str = Field(description="URL to the generated HTML tearsheet")
    message: str
    price_data: Optional[Dict[str, Any]] = Field(default=None, description="Adjusted close price data for CSV download")


class MultiSecurityCompareRequest(BaseModel):
    """Request model for multi-security comparison"""
    symbols: List[str] = Field(description="List of stock symbols (max 6)", min_length=1, max_length=6)
    benchmark: str = Field(default="SPY", description="Benchmark symbol")
    start_date: str = Field(description="Start date (YYYY-MM-DD)")
    end_date: str = Field(description="End date (YYYY-MM-DD)")
    risk_free_rate: float = Field(default=0.0, description="Risk-free rate as decimal (e.g., 0.045 for 4.5%)")
    omega_threshold: Optional[float] = Field(default=None, description="Optional Omega ratio threshold as decimal (e.g., 0.05 for 5%). If not provided, uses default 0%")
    var_confidence: Optional[float] = Field(default=None, description="Optional VaR/CVaR confidence level as decimal (e.g., 0.99 for 99%). If not provided, uses default 0.95")
    tail_cutoff: Optional[float] = Field(default=None, description="Optional Tail Ratio percentile as decimal (e.g., 0.99 for 99%). If not provided, uses default 0.95")


class TimeSeriesData(BaseModel):
    """Time series data for charts"""
    dates: List[str] = Field(description="ISO date strings")
    cumulative_returns: List[float] = Field(description="Cumulative returns (1 = 100%)")
    drawdowns: List[float] = Field(description="Drawdown values (negative, 0 to -1)")


class SymbolMetrics(BaseModel):
    """Metrics for a single symbol"""
    symbol: str
    success: bool
    trading_days: Optional[int] = None
    error: Optional[str] = None
    all_metrics: Optional[Dict[str, Any]] = None
    categorized: Optional[Dict[str, Any]] = None
    time_series: Optional[TimeSeriesData] = None


class MultiSecurityCompareResponse(BaseModel):
    """Response model for multi-security comparison"""
    success: bool
    benchmark: str
    start_date: str
    end_date: str
    risk_free_rate: float
    symbols: List[SymbolMetrics]
    benchmark_metrics: Optional[SymbolMetrics] = Field(default=None, description="Metrics for the benchmark itself")
    categories: List[Dict[str, str]] = Field(description="List of category keys and titles for tabs")
    message: str
    price_data: Optional[Dict[str, Any]] = Field(default=None, description="Adjusted close price data for CSV download")


# =============================================================================
# PORTFOLIO MODELS
# =============================================================================

class HoldingInput(BaseModel):
    """Single holding in a portfolio"""
    symbol: str = Field(description="Stock ticker symbol")
    weight: float = Field(description="Weight as decimal (0.0-1.0), e.g., 0.40 for 40%", ge=0.0, le=1.0)


class PortfolioInput(BaseModel):
    """Portfolio definition with name and holdings"""
    name: str = Field(description="Portfolio name (max 6 chars)", max_length=6)
    holdings: List[HoldingInput] = Field(description="List of holdings (1-10)", min_length=1, max_length=10)


class PortfolioCompareRequest(BaseModel):
    """Request model for portfolio comparison"""
    portfolios: List[PortfolioInput] = Field(description="List of portfolios (1-6)", min_length=1, max_length=6)
    benchmark: str = Field(default="SPY", description="Benchmark symbol")
    start_date: str = Field(description="Start date (YYYY-MM-DD)")
    end_date: str = Field(description="End date (YYYY-MM-DD)")
    risk_free_rate: float = Field(default=0.0, description="Risk-free rate as decimal")
    rebalance: str = Field(default="1ME", description="Rebalance frequency: 1ME (monthly), 1QE (quarterly), 1YE (yearly), or empty for none")
    omega_threshold: Optional[float] = Field(default=None, description="Optional Omega ratio threshold")
    var_confidence: Optional[float] = Field(default=None, description="Optional VaR/CVaR confidence level")
    tail_cutoff: Optional[float] = Field(default=None, description="Optional Tail Ratio percentile")


class PortfolioMetrics(BaseModel):
    """Metrics for a single portfolio"""
    name: str
    success: bool
    holdings: List[HoldingInput]
    total_weight: float
    trading_days: Optional[int] = None
    error: Optional[str] = None
    all_metrics: Optional[Dict[str, Any]] = None
    categorized: Optional[Dict[str, Any]] = None
    time_series: Optional[TimeSeriesData] = None


class PortfolioCompareResponse(BaseModel):
    """Response model for portfolio comparison"""
    success: bool
    benchmark: str
    start_date: str
    end_date: str
    risk_free_rate: float
    rebalance: str
    portfolios: List[PortfolioMetrics]
    benchmark_metrics: Optional[SymbolMetrics] = Field(default=None, description="Metrics for the benchmark")
    categories: List[Dict[str, str]] = Field(description="List of category keys and titles for tabs")
    message: str


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Health check and info endpoint"""
    return {
        "service": "QREP Backend API",
        "status": "running"
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/qpulse/analyze", response_model=QREPAnalysisResponse)
@limiter.limit(RATE_LIMIT)
@limiter.shared_limit(GLOBAL_RATE_LIMIT, scope="global", key_func=lambda *args, **kwargs: "global")
async def qpulse_analyze(
    request: Request,
    data: QREPAnalysisRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Generate QREP tearsheet (powered by QuantStats).

    Protected by API key if QPULSE_API_KEY env var is set.
    """
    # Verify API key
    verify_api_key(x_api_key)

    client_ip = get_real_client_ip(request)
    await check_concurrency(client_ip)
    try:
        symbol = validate_symbol(data.symbol)
        benchmark = validate_symbol(data.benchmark)

        rf_rate = data.risk_free_rate

        logger.info(f"[QREP] Generating tearsheet: {symbol} vs {benchmark}")
        logger.info(f"[QREP] Period: {data.start_date} to {data.end_date}")
        logger.info(f"[QREP] Risk-free rate: {rf_rate:.4f} ({rf_rate * 100:.2f}%)")

        # Parse dates
        start = pd.to_datetime(data.start_date)
        end = pd.to_datetime(data.end_date) + timedelta(days=1)

        # Download data using yfinance
        logger.info(f"[QREP] Downloading {symbol} data...")
        symbol_data = yf.download(symbol, start=start, end=end, progress=False)

        if symbol_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

        symbol_returns, symbol_prices = process_stock_data(symbol_data, symbol)

        logger.info(f"[QREP] Downloading {benchmark} data...")
        benchmark_data = yf.download(benchmark, start=start, end=end, progress=False)

        if benchmark_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {benchmark}")

        benchmark_returns, benchmark_prices = process_stock_data(benchmark_data, benchmark)

        if len(symbol_returns) == 0:
            raise HTTPException(status_code=400, detail=f"No data for {symbol} in date range")

        logger.info(f"[QREP] Data: {len(symbol_returns)} trading days")

        # Generate report filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_symbol = sanitize_for_filename(symbol)
        safe_benchmark = sanitize_for_filename(benchmark)
        report_filename = f'qrep_{safe_symbol}_vs_{safe_benchmark}_{timestamp}.html'
        report_path = os.path.join(REPORTS_DIR, report_filename)

        # Generate report using original quantstats with risk-free rate
        logger.info(f"[QREP] Generating report with original QuantStats...")
        qs.reports.html(
            symbol_returns,
            benchmark_returns,
            rf=rf_rate,
            output=report_path,
            title=f'{symbol} vs {benchmark} Tearsheet'
        )

        # Add QREP branding
        add_qrep_branding(report_path)

        # Construct URL
        report_url = f"{BASE_URL}/static/reports/{report_filename}"

        logger.info(f"[QREP] Report generated: {report_url}")

        # Build price data for CSV download
        price_df = pd.DataFrame({symbol: symbol_prices, benchmark: benchmark_prices})
        price_df = price_df.dropna(how='all').sort_index()
        price_data = {
            "dates": [d.strftime('%Y-%m-%d') for d in price_df.index],
            "symbols": {col: [round(float(v), 4) if pd.notna(v) else None for v in price_df[col]] for col in price_df.columns}
        }

        return QREPAnalysisResponse(
            success=True,
            html_url=report_url,
            message=f"Tearsheet generated for {symbol} vs {benchmark}",
            price_data=price_data
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[QREP] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")
    finally:
        await release_concurrency(client_ip)


@app.get("/qpulse/analyze", response_model=QREPAnalysisResponse)
@limiter.limit(RATE_LIMIT)
@limiter.shared_limit(GLOBAL_RATE_LIMIT, scope="global", key_func=lambda *args, **kwargs: "global")
async def qpulse_analyze_get(
    request: Request,
    symbol: str = Query(..., description="Stock symbol"),
    benchmark: str = Query(default="SPY", description="Benchmark symbol"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    risk_free_rate: float = Query(default=0.0, description="Risk-free rate as decimal"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """GET version of QREP analyze for browser/curl testing"""
    data = QREPAnalysisRequest(
        symbol=symbol,
        benchmark=benchmark,
        start_date=start_date,
        end_date=end_date,
        risk_free_rate=risk_free_rate
    )
    return await qpulse_analyze(request, data, x_api_key)


@app.get("/reports/{filename}")
async def serve_report(filename: str):
    """Serve generated report files"""
    safe_name = os.path.basename(filename)
    if safe_name != filename or '..' in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = os.path.join(REPORTS_DIR, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(file_path, media_type="text/html")


@app.post("/qpulse/compare", response_model=MultiSecurityCompareResponse)
@limiter.limit(RATE_LIMIT)
@limiter.shared_limit(GLOBAL_RATE_LIMIT, scope="global", key_func=lambda *args, **kwargs: "global")
async def qpulse_compare(
    request: Request,
    data: MultiSecurityCompareRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Compare multiple securities with full metrics.
    Returns categorized metrics for tabs UI.
    Max 6 securities.
    """
    verify_api_key(x_api_key)

    client_ip = get_real_client_ip(request)
    await check_concurrency(client_ip)
    try:
        symbols = [validate_symbol(s) for s in data.symbols]
        benchmark = validate_symbol(data.benchmark)
        rf_rate = data.risk_free_rate
        omega_threshold = data.omega_threshold
        var_confidence = data.var_confidence
        tail_cutoff = data.tail_cutoff

        logger.info(f"[Compare] Comparing: {symbols} vs {benchmark}")
        logger.info(f"[Compare] Period: {data.start_date} to {data.end_date}")
        logger.info(f"[Compare] Risk-free rate: {rf_rate:.4f}")
        if omega_threshold is not None:
            logger.info(f"[Compare] Omega threshold: {omega_threshold:.4f} ({omega_threshold*100:.2f}%)")
        if var_confidence is not None:
            logger.info(f"[Compare] VaR confidence: {var_confidence:.4f} ({var_confidence*100:.0f}%)")
        if tail_cutoff is not None:
            logger.info(f"[Compare] Tail cutoff: {tail_cutoff:.4f} ({tail_cutoff*100:.0f}%)")

        # Parse dates
        start = pd.to_datetime(data.start_date)
        end = pd.to_datetime(data.end_date) + timedelta(days=1)

        # Download benchmark data first
        logger.info(f"[Compare] Downloading {benchmark} data...")
        benchmark_data = yf.download(benchmark, start=start, end=end, progress=False)

        if benchmark_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for benchmark {benchmark}")

        benchmark_returns, benchmark_prices = process_stock_data(benchmark_data, benchmark)
        logger.info(f"[Compare] Benchmark {benchmark}: {len(benchmark_returns)} trading days")

        # Process each symbol
        symbol_results = []
        all_prices = {benchmark: benchmark_prices}  # Collect close prices for CSV download
        first_benchmark_metrics = None  # Will capture from first successful QuantStats call

        for symbol in symbols:
            logger.info(f"[Compare] Processing {symbol}...")
            try:
                # Download symbol data
                symbol_data = yf.download(symbol, start=start, end=end, progress=False)

                if symbol_data.empty:
                    symbol_results.append(SymbolMetrics(
                        symbol=symbol,
                        success=False,
                        error=f"No data found for {symbol}"
                    ))
                    continue

                symbol_returns, symbol_prices = process_stock_data(symbol_data, symbol)

                if len(symbol_returns) == 0:
                    symbol_results.append(SymbolMetrics(
                        symbol=symbol,
                        success=False,
                        error=f"No trading data for {symbol} in date range"
                    ))
                    continue

                all_prices[symbol] = symbol_prices

                # Get metrics (returns BOTH strategy AND benchmark from native QuantStats)
                metrics_data = get_metrics_for_symbol(symbol_returns, benchmark_returns, rf_rate, omega_threshold, var_confidence, tail_cutoff)

                # Capture benchmark metrics from the first successful call (all calls return same benchmark data)
                if first_benchmark_metrics is None and metrics_data.get('benchmark_all_metrics'):
                    first_benchmark_metrics = {
                        'all_metrics': metrics_data['benchmark_all_metrics'],
                        'categorized': metrics_data['benchmark_categorized']
                    }
                    logger.info(f"[Compare] Captured benchmark metrics from {symbol}'s QuantStats call")

                # Compute time series data for charts
                # Cumulative returns: (1 + r1) * (1 + r2) * ... - 1
                cumulative = (1 + symbol_returns).cumprod()
                # Drawdown: current value / running max - 1
                running_max = cumulative.cummax()
                drawdown = (cumulative / running_max) - 1

                # Convert to lists for JSON serialization
                # Sample every Nth point if too many data points (keep max ~250 points for charts)
                n_points = len(symbol_returns)
                sample_rate = max(1, n_points // 250)

                dates_list = [d.strftime('%Y-%m-%d') for d in symbol_returns.index[::sample_rate]]
                cumulative_list = [float(v) - 1 for v in cumulative.values[::sample_rate]]  # Convert to percentage (0 = start)
                drawdown_list = [float(v) for v in drawdown.values[::sample_rate]]

                time_series = TimeSeriesData(
                    dates=dates_list,
                    cumulative_returns=cumulative_list,
                    drawdowns=drawdown_list
                )

                symbol_results.append(SymbolMetrics(
                    symbol=symbol,
                    success=True,
                    trading_days=len(symbol_returns),
                    all_metrics=metrics_data['all_metrics'],
                    categorized=metrics_data['categorized'],
                    time_series=time_series
                ))

                logger.info(f"[Compare] {symbol}: {len(symbol_returns)} trading days, {len(dates_list)} chart points")

            except Exception as e:
                logger.error(f"[Compare] Error processing {symbol}: {e}")
                symbol_results.append(SymbolMetrics(
                    symbol=symbol,
                    success=False,
                    error=f"Failed to process data"
                ))

        # Define categories for tabs
        categories = [
            {'key': 'overview', 'title': 'Overview'},
            {'key': 'ratios', 'title': 'Risk-Adjusted Ratios'},
            {'key': 'drawdown', 'title': 'Drawdown Analysis'},
            {'key': 'returns', 'title': 'Period Returns'},
            {'key': 'risk', 'title': 'Risk Metrics'},
            {'key': 'winloss', 'title': 'Win/Loss Analysis'},
            {'key': 'benchmark', 'title': 'Benchmark Comparison'},
        ]

        # Build benchmark_metrics_result from the captured native QuantStats data
        benchmark_metrics_result = None
        if first_benchmark_metrics:
            try:
                # Compute time series data for benchmark chart
                bench_cumulative = (1 + benchmark_returns).cumprod()
                bench_running_max = bench_cumulative.cummax()
                bench_drawdown = (bench_cumulative / bench_running_max) - 1

                n_points = len(benchmark_returns)
                sample_rate = max(1, n_points // 250)

                bench_dates_list = [d.strftime('%Y-%m-%d') for d in benchmark_returns.index[::sample_rate]]
                bench_cumulative_list = [float(v) - 1 for v in bench_cumulative.values[::sample_rate]]
                bench_drawdown_list = [float(v) for v in bench_drawdown.values[::sample_rate]]

                benchmark_time_series = TimeSeriesData(
                    dates=bench_dates_list,
                    cumulative_returns=bench_cumulative_list,
                    drawdowns=bench_drawdown_list
                )

                benchmark_metrics_result = SymbolMetrics(
                    symbol=benchmark,
                    success=True,
                    trading_days=len(benchmark_returns),
                    all_metrics=first_benchmark_metrics['all_metrics'],
                    categorized=first_benchmark_metrics['categorized'],
                    time_series=benchmark_time_series
                )
                logger.info(f"[Compare] Benchmark {benchmark}: using native QuantStats metrics")
            except Exception as e:
                logger.error(f"[Compare] Error building benchmark result: {e}")
                benchmark_metrics_result = None

        successful_count = sum(1 for s in symbol_results if s.success)

        # Build price data for CSV download
        price_df = pd.DataFrame(all_prices)
        price_df = price_df.dropna(how='all').sort_index()
        price_data = {
            "dates": [d.strftime('%Y-%m-%d') for d in price_df.index],
            "symbols": {col: [round(float(v), 4) if pd.notna(v) else None for v in price_df[col]] for col in price_df.columns}
        }

        return MultiSecurityCompareResponse(
            success=successful_count > 0,
            benchmark=benchmark,
            start_date=data.start_date,
            end_date=data.end_date,
            risk_free_rate=rf_rate,
            symbols=symbol_results,
            benchmark_metrics=benchmark_metrics_result,
            categories=categories,
            message=f"Compared {successful_count}/{len(symbols)} securities successfully",
            price_data=price_data
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Compare] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")
    finally:
        await release_concurrency(client_ip)


# =============================================================================
# COMPARE EXPORT ENDPOINTS - HTML & PDF
# =============================================================================

def generate_comparison_html(
    symbols_data: List[Dict],
    benchmark: str,
    start_date: str,
    end_date: str,
    risk_free_rate: float,
    categories: List[Dict]
) -> str:
    """Generate a styled HTML report for multi-security comparison"""
    import io
    import base64
    import matplotlib.pyplot as plt

    # Get successful symbols
    successful_symbols = [s for s in symbols_data if s.get('success', False)]
    symbol_names = [s['symbol'] for s in successful_symbols]

    # Chart colors matching frontend
    chart_colors = ['#0d9488', '#6366f1', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']

    def fig_to_base64(fig):
        """Convert matplotlib figure to base64 string"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return f"data:image/png;base64,{img_str}"

    # Generate cumulative returns chart
    cumulative_chart = ""
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        for idx, sym_data in enumerate(successful_symbols):
            if sym_data.get('time_series') and sym_data['time_series'].get('dates'):
                dates = pd.to_datetime(sym_data['time_series']['dates'])
                cum_returns = [r * 100 for r in sym_data['time_series']['cumulative_returns']]
                ax.plot(dates, cum_returns, label=sym_data['symbol'],
                       color=chart_colors[idx % len(chart_colors)], linewidth=1.5)
        ax.axhline(y=0, color='#94a3b8', linestyle='--', linewidth=0.8)
        ax.set_ylabel('Cumulative Return (%)')
        ax.set_title('Cumulative Returns Comparison')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        cumulative_chart = fig_to_base64(fig)
    except Exception as e:
        logger.warning(f"Failed to generate cumulative chart: {e}")

    # Generate drawdown chart
    drawdown_chart = ""
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        for idx, sym_data in enumerate(successful_symbols):
            if sym_data.get('time_series') and sym_data['time_series'].get('dates'):
                dates = pd.to_datetime(sym_data['time_series']['dates'])
                drawdowns = [r * 100 for r in sym_data['time_series']['drawdowns']]
                ax.fill_between(dates, drawdowns, 0, alpha=0.3,
                              color=chart_colors[idx % len(chart_colors)])
                ax.plot(dates, drawdowns, label=sym_data['symbol'],
                       color=chart_colors[idx % len(chart_colors)], linewidth=1.5)
        ax.axhline(y=0, color='#94a3b8', linestyle='--', linewidth=0.8)
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown Comparison')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        drawdown_chart = fig_to_base64(fig)
    except Exception as e:
        logger.warning(f"Failed to generate drawdown chart: {e}")

    # Generate key ratios bar chart
    ratios_chart = ""
    try:
        ratio_metrics = ['Sharpe', 'Sortino', 'Calmar']
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(ratio_metrics))
        width = 0.8 / len(successful_symbols)

        for idx, sym_data in enumerate(successful_symbols):
            values = []
            for metric in ratio_metrics:
                val = sym_data.get('all_metrics', {}).get(metric, 0)
                values.append(val if isinstance(val, (int, float)) and np.isfinite(val) else 0)
            offset = (idx - len(successful_symbols)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=sym_data['symbol'],
                  color=chart_colors[idx % len(chart_colors)])

        ax.axhline(y=0, color='#94a3b8', linestyle='--', linewidth=0.8)
        ax.set_ylabel('Ratio Value')
        ax.set_title('Key Risk-Adjusted Ratios')
        ax.set_xticks(x)
        ax.set_xticklabels(ratio_metrics)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        ratios_chart = fig_to_base64(fig)
    except Exception as e:
        logger.warning(f"Failed to generate ratios chart: {e}")

    # Build metrics tables HTML
    def format_value(val):
        if val is None:
            return '-'
        if isinstance(val, (int, float)):
            if not np.isfinite(val):
                return '-'
            if abs(val) >= 1000:
                return f"{val:,.2f}"
            return f"{val:.4f}"
        return str(val)

    tables_html = ""
    for cat in categories:
        cat_key = cat['key']
        cat_title = cat['title']

        # Collect all metrics for this category
        all_metrics = set()
        for sym_data in successful_symbols:
            if sym_data.get('categorized', {}).get(cat_key, {}).get('metrics'):
                all_metrics.update(sym_data['categorized'][cat_key]['metrics'].keys())

        if not all_metrics:
            continue

        metric_list = sorted(list(all_metrics))

        # Build table
        table_rows = ""
        for idx, metric in enumerate(metric_list):
            row_class = 'even' if idx % 2 == 0 else 'odd'
            cells = f'<td class="metric-name">{metric}</td>'
            for sym_data in successful_symbols:
                val = sym_data.get('categorized', {}).get(cat_key, {}).get('metrics', {}).get(metric)
                cells += f'<td class="metric-value">{format_value(val)}</td>'
            table_rows += f'<tr class="{row_class}">{cells}</tr>'

        header_cells = '<th>Metric</th>'
        for sym_data in successful_symbols:
            header_cells += f'<th>{sym_data["symbol"]}</th>'

        tables_html += f'''
        <div class="category-section">
            <h2>{cat_title}</h2>
            <table class="metrics-table">
                <thead><tr>{header_cells}</tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>
        '''

    # Build final HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Security Comparison Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            color: #000000;
            background: #ffffff;
            line-height: 1.5;
            padding: 20px;
        }}
        .header {{
            border-bottom: 2px solid #0d9488;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            font-size: 24px;
            color: #0d9488;
            margin-bottom: 8px;
        }}
        .header .meta {{
            font-size: 14px;
            color: #000000;
        }}
        .header .meta span {{
            margin-right: 20px;
        }}
        .charts-section {{
            margin-bottom: 30px;
        }}
        .chart-container {{
            margin-bottom: 20px;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
        }}
        .category-section {{
            margin-bottom: 30px;
            page-break-inside: avoid;
        }}
        .category-section h2 {{
            font-size: 18px;
            color: #0d9488;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 8px;
            margin-bottom: 12px;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        .metrics-table th {{
            background: #f1f5f9;
            padding: 10px 12px;
            text-align: right;
            font-weight: 600;
            border-bottom: 2px solid #cbd5e1;
        }}
        .metrics-table th:first-child {{
            text-align: left;
        }}
        .metrics-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid #e2e8f0;
        }}
        .metrics-table tr.even {{
            background: #ffffff;
        }}
        .metrics-table tr.odd {{
            background: #f8fafc;
        }}
        .metric-name {{
            font-weight: 500;
            text-align: left;
        }}
        .metric-value {{
            text-align: right;
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #e2e8f0;
            font-size: 12px;
            color: #64748b;
            text-align: center;
        }}
        @media print {{
            body {{ padding: 10px; }}
            .chart-container {{ page-break-inside: avoid; }}
            .category-section {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Multi-Security Comparison Report</h1>
        <div class="meta">
            <span><strong>Securities:</strong> {', '.join(symbol_names)}</span>
            <span><strong>Benchmark:</strong> {benchmark}</span>
            <span><strong>Period:</strong> {start_date} to {end_date}</span>
            <span><strong>Risk-Free Rate:</strong> {risk_free_rate*100:.2f}%</span>
        </div>
    </div>

    <div class="charts-section">
        <h2 style="font-size: 18px; color: #0d9488; margin-bottom: 15px;">Performance Charts</h2>
        {f'<div class="chart-container"><img src="{cumulative_chart}" alt="Cumulative Returns"></div>' if cumulative_chart else ''}
        {f'<div class="chart-container"><img src="{drawdown_chart}" alt="Drawdown"></div>' if drawdown_chart else ''}
        {f'<div class="chart-container"><img src="{ratios_chart}" alt="Key Ratios"></div>' if ratios_chart else ''}
    </div>

    {tables_html}

    <div class="footer">
        Generated by QREP (Powered by QuantStats) | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>'''

    return html


def generate_comparison_pdf_reportlab(
    symbols_data: List[Dict],
    benchmark: str,
    start_date: str,
    end_date: str,
    risk_free_rate: float,
    categories: List[Dict],
    output_path: str
) -> None:
    """Generate a PDF report using ReportLab for multi-security comparison"""

    # Get successful symbols
    successful_symbols = [s for s in symbols_data if s.get('success', False)]
    symbol_names = [s['symbol'] for s in successful_symbols]

    # Chart colors
    chart_colors = ['#0d9488', '#6366f1', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']

    # Create document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
        title="Multi-Security Comparison Report"
    )

    # Setup styles
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#0d9488'),
        spaceAfter=6,
        spaceBefore=0
    )

    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#000000'),
        spaceAfter=12
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#0d9488'),
        spaceBefore=16,
        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#000000'),
        spaceAfter=6
    )

    # Table styles
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 4),
        ('TOPPADDING', (0, 1), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
    ])

    story = []

    # Title
    story.append(Paragraph("Multi-Security Comparison Report", title_style))
    story.append(HRFlowable(
        width="100%",
        thickness=2,
        color=colors.HexColor('#0d9488'),
        spaceBefore=0,
        spaceAfter=8
    ))

    # Metadata
    meta_text = f"Securities: {', '.join(symbol_names)} | Benchmark: {benchmark} | Period: {start_date} to {end_date}"
    story.append(Paragraph(meta_text, subtitle_style))

    # Generate and add charts
    def create_chart_image(fig, width=6.5*inch, height=3*inch):
        """Convert matplotlib figure to ReportLab Image"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        plt.close(fig)
        return RLImage(buf, width=width, height=height)

    # Cumulative Returns Chart
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        for idx, sym_data in enumerate(successful_symbols):
            if sym_data.get('time_series') and sym_data['time_series'].get('dates'):
                dates = pd.to_datetime(sym_data['time_series']['dates'])
                cum_returns = [r * 100 for r in sym_data['time_series']['cumulative_returns']]
                ax.plot(dates, cum_returns, label=sym_data['symbol'],
                       color=chart_colors[idx % len(chart_colors)], linewidth=1.5)
        ax.axhline(y=0, color='#94a3b8', linestyle='--', linewidth=0.8)
        ax.set_ylabel('Cumulative Return (%)')
        ax.set_title('Cumulative Returns Comparison', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        story.append(Paragraph("Cumulative Returns", heading_style))
        story.append(create_chart_image(fig))
        story.append(Spacer(1, 12))
    except Exception as e:
        logger.warning(f"Failed to generate cumulative chart for PDF: {e}")

    # Drawdown Chart
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        for idx, sym_data in enumerate(successful_symbols):
            if sym_data.get('time_series') and sym_data['time_series'].get('drawdowns'):
                dates = pd.to_datetime(sym_data['time_series']['dates'])
                drawdowns = [d * 100 for d in sym_data['time_series']['drawdowns']]
                ax.fill_between(dates, 0, drawdowns, alpha=0.3,
                              color=chart_colors[idx % len(chart_colors)])
                ax.plot(dates, drawdowns, label=sym_data['symbol'],
                       color=chart_colors[idx % len(chart_colors)], linewidth=1)
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown Comparison', fontsize=11, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        story.append(Paragraph("Drawdown Analysis", heading_style))
        story.append(create_chart_image(fig))
        story.append(Spacer(1, 12))
    except Exception as e:
        logger.warning(f"Failed to generate drawdown chart for PDF: {e}")

    # Key Metrics Bar Chart
    try:
        metrics_to_plot = ['CAGR', 'Sharpe', 'Max Drawdown', 'Volatility']
        fig, ax = plt.subplots(figsize=(10, 4))

        x = np.arange(len(metrics_to_plot))
        width = 0.8 / max(len(successful_symbols), 1)

        for idx, sym_data in enumerate(successful_symbols):
            values = []
            for metric in metrics_to_plot:
                val = sym_data.get('all_metrics', {}).get(metric, 0)
                if isinstance(val, str):
                    val = float(val.replace('%', '').replace(',', '')) if val not in ['N/A', '-'] else 0
                values.append(val)

            ax.bar(x + idx * width, values, width, label=sym_data['symbol'],
                  color=chart_colors[idx % len(chart_colors)])

        ax.set_ylabel('Value')
        ax.set_title('Key Metrics Comparison', fontsize=11, fontweight='bold')
        ax.set_xticks(x + width * (len(successful_symbols) - 1) / 2)
        ax.set_xticklabels(metrics_to_plot)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        story.append(Paragraph("Key Metrics", heading_style))
        story.append(create_chart_image(fig))
        story.append(Spacer(1, 12))
    except Exception as e:
        logger.warning(f"Failed to generate metrics chart for PDF: {e}")

    # Add metrics tables by category
    for category in categories:
        cat_key = category['key']
        cat_title = category['title']

        story.append(Paragraph(cat_title, heading_style))

        # Build table data
        table_data = [['Metric'] + symbol_names]

        # Get metrics for this category from first successful symbol
        if successful_symbols:
            cat_metrics = successful_symbols[0].get('categorized', {}).get(cat_key, {})
            for metric_name in cat_metrics.keys():
                row = [metric_name]
                for sym_data in successful_symbols:
                    value = sym_data.get('categorized', {}).get(cat_key, {}).get(metric_name, 'N/A')
                    row.append(str(value))
                table_data.append(row)

        if len(table_data) > 1:
            # Calculate column widths
            num_cols = len(table_data[0])
            metric_col_width = 2 * inch
            data_col_width = (6.5 * inch - metric_col_width) / max(num_cols - 1, 1)
            col_widths = [metric_col_width] + [data_col_width] * (num_cols - 1)

            table = Table(table_data, colWidths=col_widths)
            table.setStyle(table_style)
            story.append(table)
            story.append(Spacer(1, 12))

    # Footer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(
        width="100%",
        thickness=1,
        color=colors.HexColor('#e2e8f0'),
        spaceBefore=0,
        spaceAfter=8
    ))
    footer_text = f"Generated by QREP (Powered by QuantStats) | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#64748b'),
        alignment=1  # Center
    )
    story.append(Paragraph(footer_text, footer_style))

    # Build PDF
    doc.build(story)
    logger.info(f"[PDF] Generated comparison report: {output_path}")


class CompareExportRequest(BaseModel):
    """Request model for comparison export"""
    symbols: List[str] = Field(description="List of symbols to compare")
    benchmark: str = Field(default="SPY", description="Benchmark symbol")
    start_date: str = Field(description="Start date YYYY-MM-DD")
    end_date: str = Field(description="End date YYYY-MM-DD")
    risk_free_rate: float = Field(default=0.0, description="Risk-free rate as decimal")
    omega_threshold: Optional[float] = Field(default=None, description="Optional Omega ratio threshold as decimal")
    var_confidence: Optional[float] = Field(default=None, description="Optional VaR/CVaR confidence level as decimal")
    tail_cutoff: Optional[float] = Field(default=None, description="Optional Tail Ratio percentile as decimal")
    format: str = Field(default="html", description="Export format: html or pdf")


@app.post("/qpulse/compare/export")
@limiter.limit(RATE_LIMIT)
@limiter.shared_limit(GLOBAL_RATE_LIMIT, scope="global", key_func=lambda *args, **kwargs: "global")
async def export_comparison(
    request: Request,
    data: CompareExportRequest,
    x_api_key: str = Header(None, alias="X-API-Key")
):
    """
    Export comparison results as HTML or PDF
    """
    verify_api_key(x_api_key)

    client_ip = get_real_client_ip(request)
    await check_concurrency(client_ip)

    logger.info(f"[Export] Exporting comparison: {data.symbols} as {data.format}")

    try:
        # Re-run the comparison logic inline (to avoid Request dependency)
        symbols = [validate_symbol(s) for s in data.symbols]
        benchmark = validate_symbol(data.benchmark)
        rf_rate = data.risk_free_rate
        omega_threshold = data.omega_threshold
        var_confidence = data.var_confidence
        tail_cutoff = data.tail_cutoff

        logger.info(f"[Export] Processing comparison for export: {symbols}")

        # Parse dates
        start = pd.to_datetime(data.start_date)
        end = pd.to_datetime(data.end_date) + timedelta(days=1)

        # Download benchmark data
        benchmark_data = yf.download(benchmark, start=start, end=end, progress=False)
        if benchmark_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for benchmark {benchmark}")

        benchmark_returns, _ = process_stock_data(benchmark_data, benchmark)

        # Process each symbol
        symbol_results = []
        for symbol in symbols:
            try:
                symbol_data = yf.download(symbol, start=start, end=end, progress=False)
                if symbol_data.empty:
                    symbol_results.append({
                        'symbol': symbol,
                        'success': False,
                        'error': f"No data found for {symbol}"
                    })
                    continue

                symbol_returns, _ = process_stock_data(symbol_data, symbol)
                if len(symbol_returns) == 0:
                    symbol_results.append({
                        'symbol': symbol,
                        'success': False,
                        'error': f"No trading data for {symbol}"
                    })
                    continue

                # Get metrics
                metrics_data = get_metrics_for_symbol(symbol_returns, benchmark_returns, rf_rate, omega_threshold, var_confidence, tail_cutoff)

                # Compute time series
                cumulative = (1 + symbol_returns).cumprod()
                running_max = cumulative.cummax()
                drawdown = (cumulative / running_max) - 1

                n_points = len(symbol_returns)
                sample_rate = max(1, n_points // 250)

                symbol_results.append({
                    'symbol': symbol,
                    'success': True,
                    'trading_days': len(symbol_returns),
                    'all_metrics': metrics_data['all_metrics'],
                    'categorized': metrics_data['categorized'],
                    'time_series': {
                        'dates': [d.strftime('%Y-%m-%d') for d in symbol_returns.index[::sample_rate]],
                        'cumulative_returns': [float(v) - 1 for v in cumulative.values[::sample_rate]],
                        'drawdowns': [float(v) for v in drawdown.values[::sample_rate]]
                    }
                })

            except Exception as e:
                logger.error(f"[Export] Error processing {symbol}: {e}")
                symbol_results.append({
                    'symbol': symbol,
                    'success': False,
                    'error': str(e)
                })

        # Define categories
        categories = [
            {'key': 'overview', 'title': 'Overview'},
            {'key': 'ratios', 'title': 'Risk-Adjusted Ratios'},
            {'key': 'drawdown', 'title': 'Drawdown Analysis'},
            {'key': 'returns', 'title': 'Period Returns'},
            {'key': 'risk', 'title': 'Risk Metrics'},
            {'key': 'winloss', 'title': 'Win/Loss Analysis'},
            {'key': 'benchmark', 'title': 'Benchmark Comparison'},
        ]

        symbols_data = symbol_results

        # Generate HTML
        html_content = generate_comparison_html(
            symbols_data=symbols_data,
            benchmark=benchmark,
            start_date=data.start_date,
            end_date=data.end_date,
            risk_free_rate=rf_rate,
            categories=categories
        )

        if data.format.lower() == 'pdf':
            # Generate PDF using ReportLab
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbols_str = "_".join(data.symbols[:3])  # First 3 symbols for filename
            filename = f"comparison_{symbols_str}_{timestamp}.pdf"
            filepath = os.path.join(REPORTS_DIR, filename)

            # Generate PDF with ReportLab
            generate_comparison_pdf_reportlab(
                symbols_data=symbols_data,
                benchmark=benchmark,
                start_date=data.start_date,
                end_date=data.end_date,
                risk_free_rate=rf_rate,
                categories=categories,
                output_path=filepath
            )

            logger.info(f"[Export] PDF generated with ReportLab: {filename}")

            return FileResponse(
                path=filepath,
                filename=filename,
                media_type='application/pdf'
            )
        else:
            # Return HTML directly
            return HTMLResponse(content=html_content)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Export] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")
    finally:
        await release_concurrency(client_ip)


# =============================================================================
# PORTFOLIO ENDPOINT
# =============================================================================

@app.post("/qpulse/portfolio", response_model=PortfolioCompareResponse)
@limiter.limit(RATE_LIMIT)
@limiter.shared_limit(GLOBAL_RATE_LIMIT, scope="global", key_func=lambda *args, **kwargs: "global")
async def portfolio_compare(
    request: Request,
    data: PortfolioCompareRequest,
    x_api_key: str = Header(None, alias="X-API-Key")
):
    """
    Compare multiple portfolios with weighted holdings.
    Each portfolio can have up to 10 holdings with custom weights.
    Returns all QuantStats metrics for each portfolio.
    """
    verify_api_key(x_api_key)

    client_ip = get_real_client_ip(request)
    await check_concurrency(client_ip)
    try:
        benchmark = validate_symbol(data.benchmark)
        rf_rate = data.risk_free_rate
        rebalance = data.rebalance if data.rebalance else None
        omega_threshold = data.omega_threshold
        var_confidence = data.var_confidence
        tail_cutoff = data.tail_cutoff

        logger.info(f"[Portfolio] Comparing {len(data.portfolios)} portfolios vs {benchmark}")
        logger.info(f"[Portfolio] Period: {data.start_date} to {data.end_date}")
        logger.info(f"[Portfolio] Rebalance: {rebalance or 'None (buy-and-hold)'}")

        # Parse dates
        start = pd.to_datetime(data.start_date)
        end = pd.to_datetime(data.end_date) + timedelta(days=1)

        # Collect all unique symbols across all portfolios
        all_symbols = set()
        for portfolio in data.portfolios:
            for holding in portfolio.holdings:
                all_symbols.add(validate_symbol(holding.symbol))
        all_symbols.add(benchmark)
        all_symbols_list = list(all_symbols)

        logger.info(f"[Portfolio] Downloading data for {len(all_symbols_list)} unique symbols...")

        # Download each symbol individually (more reliable than multi-symbol download)
        returns_dict = {}
        for symbol in all_symbols_list:
            logger.info(f"[Portfolio] Downloading {symbol}...")
            symbol_data = yf.download(symbol, start=start, end=end, progress=False)
            if symbol_data.empty:
                logger.warning(f"[Portfolio] No data for {symbol}")
                continue
            symbol_returns, _ = process_stock_data(symbol_data, symbol)
            if len(symbol_returns) > 0:
                returns_dict[symbol] = symbol_returns

        if not returns_dict:
            raise HTTPException(status_code=404, detail="No data found for the specified symbols")

        # Build returns DataFrame
        returns_df = pd.DataFrame(returns_dict)
        logger.info(f"[Portfolio] Returns calculated: {len(returns_df)} trading days, {len(returns_df.columns)} symbols")

        # Get benchmark returns
        if benchmark not in returns_df.columns:
            raise HTTPException(status_code=404, detail=f"No data found for benchmark {benchmark}")
        benchmark_returns = returns_df[benchmark].dropna()

        # Process each portfolio
        portfolio_results = []
        first_benchmark_metrics = None

        for portfolio in data.portfolios:
            portfolio_name = portfolio.name.strip()[:6]  # Max 6 chars
            logger.info(f"[Portfolio] Processing portfolio: {portfolio_name}")

            try:
                # Build weights dict and validate
                ticker_weights = {}
                total_weight = 0.0
                holdings_list = []

                for holding in portfolio.holdings:
                    symbol = validate_symbol(holding.symbol)
                    weight = holding.weight
                    ticker_weights[symbol] = weight
                    total_weight += weight
                    holdings_list.append(HoldingInput(symbol=symbol, weight=weight))

                # Validate weights sum to ~1.0 (allow 0.99-1.01)
                if not (0.99 <= total_weight <= 1.01):
                    portfolio_results.append(PortfolioMetrics(
                        name=portfolio_name,
                        success=False,
                        holdings=holdings_list,
                        total_weight=round(total_weight, 4),
                        error=f"Weights must sum to 100% (got {total_weight*100:.1f}%)"
                    ))
                    continue

                # Check all symbols have data
                missing_symbols = [s for s in ticker_weights.keys() if s not in returns_df.columns]
                if missing_symbols:
                    portfolio_results.append(PortfolioMetrics(
                        name=portfolio_name,
                        success=False,
                        holdings=holdings_list,
                        total_weight=round(total_weight, 4),
                        error=f"No data for symbols: {', '.join(missing_symbols)}"
                    ))
                    continue

                # Calculate portfolio returns (weighted sum)
                portfolio_returns = pd.Series(0.0, index=returns_df.index)
                for symbol, weight in ticker_weights.items():
                    symbol_returns = returns_df[symbol].fillna(0)
                    portfolio_returns = portfolio_returns + (symbol_returns * weight)

                # Apply rebalancing if specified (resample and recalculate)
                # Note: For simplicity, we're using daily weighted returns
                # Full rebalancing would require cumulative tracking and resetting weights

                portfolio_returns = portfolio_returns.dropna()

                if len(portfolio_returns) == 0:
                    portfolio_results.append(PortfolioMetrics(
                        name=portfolio_name,
                        success=False,
                        holdings=holdings_list,
                        total_weight=round(total_weight, 4),
                        error="No valid return data for portfolio"
                    ))
                    continue

                # Get metrics using existing function
                metrics_data = get_metrics_for_symbol(
                    portfolio_returns, benchmark_returns, rf_rate,
                    omega_threshold, var_confidence, tail_cutoff
                )

                # Capture benchmark metrics from first successful call
                if first_benchmark_metrics is None and metrics_data.get('benchmark_all_metrics'):
                    first_benchmark_metrics = {
                        'all_metrics': metrics_data['benchmark_all_metrics'],
                        'categorized': metrics_data['benchmark_categorized']
                    }

                # Compute time series data
                cumulative = (1 + portfolio_returns).cumprod()
                running_max = cumulative.cummax()
                drawdown = (cumulative / running_max) - 1

                # Sample for charts
                n_points = len(portfolio_returns)
                sample_rate = max(1, n_points // 250)

                portfolio_results.append(PortfolioMetrics(
                    name=portfolio_name,
                    success=True,
                    holdings=holdings_list,
                    total_weight=round(total_weight, 4),
                    trading_days=len(portfolio_returns),
                    all_metrics=metrics_data['all_metrics'],
                    categorized=metrics_data['categorized'],
                    time_series=TimeSeriesData(
                        dates=[d.strftime('%Y-%m-%d') for d in portfolio_returns.index[::sample_rate]],
                        cumulative_returns=[float(v) - 1 for v in cumulative.values[::sample_rate]],
                        drawdowns=[float(v) for v in drawdown.values[::sample_rate]]
                    )
                ))

                logger.info(f"[Portfolio] {portfolio_name}: {len(portfolio_returns)} trading days, {len(metrics_data['all_metrics'])} metrics")

            except Exception as e:
                logger.error(f"[Portfolio] Error processing {portfolio_name}: {e}")
                portfolio_results.append(PortfolioMetrics(
                    name=portfolio_name,
                    success=False,
                    holdings=[HoldingInput(symbol=h.symbol, weight=h.weight) for h in portfolio.holdings],
                    total_weight=0.0,
                    error=f"Failed to process data"
                ))

        # Build benchmark metrics response
        benchmark_metrics_response = None
        if first_benchmark_metrics:
            # Calculate benchmark time series
            bench_cumulative = (1 + benchmark_returns).cumprod()
            bench_running_max = bench_cumulative.cummax()
            bench_drawdown = (bench_cumulative / bench_running_max) - 1
            n_points = len(benchmark_returns)
            sample_rate = max(1, n_points // 250)

            benchmark_metrics_response = SymbolMetrics(
                symbol=benchmark,
                success=True,
                trading_days=len(benchmark_returns),
                all_metrics=first_benchmark_metrics['all_metrics'],
                categorized=first_benchmark_metrics['categorized'],
                time_series=TimeSeriesData(
                    dates=[d.strftime('%Y-%m-%d') for d in benchmark_returns.index[::sample_rate]],
                    cumulative_returns=[float(v) - 1 for v in bench_cumulative.values[::sample_rate]],
                    drawdowns=[float(v) for v in bench_drawdown.values[::sample_rate]]
                )
            )

        # Define categories (same as compare endpoint)
        categories = [
            {'key': 'overview', 'title': 'Overview'},
            {'key': 'ratios', 'title': 'Risk-Adjusted Ratios'},
            {'key': 'drawdown', 'title': 'Drawdown Analysis'},
            {'key': 'returns', 'title': 'Period Returns'},
            {'key': 'risk', 'title': 'Risk Metrics'},
            {'key': 'winloss', 'title': 'Win/Loss Analysis'},
            {'key': 'benchmark', 'title': 'Benchmark Comparison'},
        ]

        successful_count = sum(1 for p in portfolio_results if p.success)
        message = f"Analyzed {successful_count}/{len(portfolio_results)} portfolios successfully"

        return PortfolioCompareResponse(
            success=successful_count > 0,
            benchmark=benchmark,
            start_date=data.start_date,
            end_date=data.end_date,
            risk_free_rate=rf_rate,
            rebalance=rebalance or "none",
            portfolios=portfolio_results,
            benchmark_metrics=benchmark_metrics_response,
            categories=categories,
            message=message
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Portfolio] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")
    finally:
        await release_concurrency(client_ip)


# =============================================================================
# MCP COMPARE ENDPOINT (no API key, slim response for LLM consumption)
# =============================================================================

class MCPCompareSymbolMetrics(BaseModel):
    """Slim metrics for MCP - only all_metrics, no charts/time_series"""
    symbol: str
    success: bool
    trading_days: Optional[int] = None
    error: Optional[str] = None
    all_metrics: Optional[Dict[str, Any]] = Field(default=None, description="81 key-value portfolio metric pairs (powered by QuantStats)")


class MCPCompareResponse(BaseModel):
    """Slim response for MCP compare endpoint - metrics only, no chart data"""
    success: bool
    benchmark: str
    start_date: str
    end_date: str
    risk_free_rate: float
    symbols: List[MCPCompareSymbolMetrics]
    benchmark_metrics: Optional[MCPCompareSymbolMetrics] = Field(default=None, description="Metrics for the benchmark")
    message: str


@app.get("/mcp/compare", response_model=MCPCompareResponse, operation_id="mcp_compare_securities")
@limiter.limit(RATE_LIMIT)
@limiter.shared_limit(GLOBAL_RATE_LIMIT, scope="global", key_func=lambda *args, **kwargs: "global")
async def mcp_compare(
    request: Request,
    symbols: str = Query(description="Comma-separated Yahoo Finance ticker symbols, max 6 (e.g. AAPL,MSFT,GOOG)"),
    benchmark: str = Query(default="SPY", description="Benchmark symbol. Can be any valid Yahoo Finance ticker."),
    start_date: str = Query(description="Start date YYYY-MM-DD. Should be at least 6 months before end_date for meaningful analysis."),
    end_date: str = Query(description="End date YYYY-MM-DD. Must not be in the future."),
    risk_free_rate: float = Query(default=0.0, description="Risk-free rate as decimal (e.g. 0.045 for 4.5%). Used in Sharpe, Sortino, Treynor, and other risk-adjusted calculations."),
):
    """
    QREP security comparison — returns 81 portfolio metrics per symbol (powered by QuantStats).
    Lightweight endpoint for LLM/MCP consumption (no time series, no chart data, no price data).
    No API key required.

    RESPONSE STRUCTURE:
    - success: boolean indicating overall success
    - benchmark: the benchmark symbol used
    - start_date / end_date: the date range used
    - risk_free_rate: the risk-free rate used
    - symbols: array of per-symbol results, each containing:
        - symbol: ticker string
        - success: boolean
        - trading_days: number of trading days in the period
        - error: null on success, error message on failure
        - all_metrics: object with 81 key-value metric pairs (see list below)
    - benchmark_metrics: same structure as a symbol entry, but for the benchmark itself.
      NOTE: Beta, Alpha, Correlation, and Treynor Ratio are null for the benchmark (cannot compare against itself).
    - message: summary string (e.g. "Compared 3/3 securities successfully")

    COMPLETE LIST OF 81 METRICS (per symbol):
    All values are decimals unless noted. Percentages as decimals (e.g. 0.31 = 31%).

    Returns & Performance:
    Cumulative Return, CAGR%, MTD, 3M, 6M, YTD, 1Y, 3Y (ann.), 5Y (ann.), 10Y (ann.), All-time (ann.),
    Best Day, Worst Day, Best Month, Worst Month, Best Year, Worst Year,
    Expected Daily, Expected Monthly, Expected Yearly,
    Avg. Return, Avg. Win, Avg. Loss, Avg. Up Month, Avg. Down Month

    Risk-Adjusted Ratios:
    Sharpe, Smart Sharpe, Prob. Sharpe Ratio,
    Sortino, Smart Sortino, Sortino/sqrt(2), Smart Sortino/sqrt(2),
    Omega, Calmar, Treynor Ratio, Information Ratio,
    Risk-Adjusted Return, Risk-Return Ratio, Ulcer Performance Index

    Drawdown:
    Max Drawdown, Max DD Date (YYYY-MM-DD), Max DD Period Start, Max DD Period End,
    Longest DD Days (integer), Avg. Drawdown, Avg. Drawdown Days (integer),
    Recovery Factor, Ulcer Index, Serenity Index

    Volatility & Distribution:
    Volatility (ann.), Skew, Kurtosis, R-squared

    Benchmark-Relative (null for the benchmark itself):
    Beta, Alpha, Correlation

    Value-at-Risk & Tail Risk:
    Daily Value-at-Risk, Expected Shortfall (cVaR),
    Risk of Ruin, Kelly Criterion,
    Tail Ratio, Outlier Win Ratio, Outlier Loss Ratio

    Win/Loss & Trade Statistics:
    Win Days %, Win Month %, Win Quarter %, Win Year %,
    Win/Loss Ratio, Profit Ratio, Payoff Ratio, Profit Factor,
    Max Consecutive Wins (integer), Max Consecutive Losses (integer),
    Gain/Pain Ratio, Gain/Pain (1M), Common Sense Ratio, CPC Index

    Time Context:
    Start Period (YYYY-MM-DD), End Period (YYYY-MM-DD),
    Risk-Free Rate (as percentage, e.g. 4.5), Time in Market (decimal, 1.0 = 100%)

    NOTES:
    1. 3Y, 5Y, 10Y, All-time annualized returns equal CAGR if the data period is shorter than those horizons.
    2. Dates in Max DD fields are strings in YYYY-MM-DD format.
    3. Integer metrics (Longest DD Days, Avg. Drawdown Days, Max Consecutive Wins/Losses) are returned as floats (e.g. 133.0).
    """
    client_ip = get_real_client_ip(request)
    await check_concurrency(client_ip)
    try:
        # Parse and validate symbols
        symbol_list = [validate_symbol(s.strip()) for s in symbols.split(",") if s.strip()]
        if not symbol_list:
            raise HTTPException(status_code=400, detail="At least one symbol required")
        if len(symbol_list) > 6:
            raise HTTPException(status_code=400, detail="Maximum 6 symbols allowed")

        bench = validate_symbol(benchmark)
        rf_rate = risk_free_rate

        logger.info(f"[MCP Compare] Comparing: {symbol_list} vs {bench}")
        logger.info(f"[MCP Compare] Period: {start_date} to {end_date}")

        # Parse dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) + timedelta(days=1)

        # Download benchmark
        benchmark_data = yf.download(bench, start=start, end=end, progress=False)
        if benchmark_data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for benchmark {bench}")

        benchmark_returns, _ = process_stock_data(benchmark_data, bench)

        # Process each symbol
        symbol_results = []
        first_benchmark_metrics = None

        for symbol in symbol_list:
            try:
                symbol_data = yf.download(symbol, start=start, end=end, progress=False)

                if symbol_data.empty:
                    symbol_results.append(MCPCompareSymbolMetrics(
                        symbol=symbol, success=False, error=f"No data found for {symbol}"
                    ))
                    continue

                symbol_returns, _ = process_stock_data(symbol_data, symbol)

                if len(symbol_returns) == 0:
                    symbol_results.append(MCPCompareSymbolMetrics(
                        symbol=symbol, success=False, error=f"No trading data for {symbol} in date range"
                    ))
                    continue

                metrics_data = get_metrics_for_symbol(symbol_returns, benchmark_returns, rf_rate)

                if first_benchmark_metrics is None and metrics_data.get('benchmark_all_metrics'):
                    first_benchmark_metrics = metrics_data['benchmark_all_metrics']

                symbol_results.append(MCPCompareSymbolMetrics(
                    symbol=symbol,
                    success=True,
                    trading_days=len(symbol_returns),
                    all_metrics=metrics_data['all_metrics']
                ))

                logger.info(f"[MCP Compare] {symbol}: {len(symbol_returns)} trading days")

            except Exception as e:
                logger.error(f"[MCP Compare] Error processing {symbol}: {e}")
                symbol_results.append(MCPCompareSymbolMetrics(
                    symbol=symbol, success=False, error="Failed to process data"
                ))

        # Build benchmark result
        benchmark_result = None
        if first_benchmark_metrics:
            benchmark_result = MCPCompareSymbolMetrics(
                symbol=bench,
                success=True,
                trading_days=len(benchmark_returns),
                all_metrics=first_benchmark_metrics
            )

        successful_count = sum(1 for s in symbol_results if s.success)

        return MCPCompareResponse(
            success=successful_count > 0,
            benchmark=bench,
            start_date=start_date,
            end_date=end_date,
            risk_free_rate=rf_rate,
            symbols=symbol_results,
            benchmark_metrics=benchmark_result,
            message=f"Compared {successful_count}/{len(symbol_list)} securities successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MCP Compare] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")
    finally:
        await release_concurrency(client_ip)


# =============================================================================
# CLEANUP ENDPOINT (for cron-job.org)
# =============================================================================

@app.post("/cleanup")
async def cleanup_reports_endpoint(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Clean up old report files. Protected by CLEANUP_API_KEY or QPULSE_API_KEY."""
    provided_key = x_api_key or ""
    valid_keys = [k for k in [CLEANUP_API_KEY, QPULSE_API_KEY] if k]
    if not valid_keys or provided_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Unauthorized")

    cleanup_old_reports(max_age_days=3)
    remaining = len(os.listdir(REPORTS_DIR)) if os.path.exists(REPORTS_DIR) else 0
    return {"status": "ok", "remaining_files": remaining}


# =============================================================================
# MCP SERVER
# =============================================================================

base_url = os.getenv("BASE_URL", "https://qpulse-api.tigzig.com")
if not base_url.startswith(("http://", "https://")):
    base_url = f"http://{base_url}"

mcp = FastApiMCP(
    app,
    name="QREP MCP API",
    description="QREP MCP server — portfolio analytics powered by QuantStats. Compare up to 6 securities with 81 metrics each.",
    include_operations=["mcp_compare_securities"],
    describe_all_responses=True,
    describe_full_response_schema=True,
    http_client=httpx.AsyncClient(
        timeout=300.0,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        base_url=base_url
    )
)

mcp.mount()
