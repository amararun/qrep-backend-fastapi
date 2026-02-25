# QREP Backend — FastAPI + MCP Server

> Built on [QuantStats](https://github.com/ranaroussi/quantstats) (Apache 2.0) by Ran Aroussi and [yfinance](https://github.com/ranaroussi/yfinance) (Apache 2.0) by Ran Aroussi. QREP is an independent project — not affiliated with or endorsed by the QuantStats project or its authors.

FastAPI backend with integrated MCP (Model Context Protocol) server for QREP Security Analytics. Computes 81 portfolio metrics per symbol using the original QuantStats library. The MCP endpoint enables AI clients (Claude, GPTs, Copilot) to query portfolio analytics as a tool.

Live endpoint: [qrep-api.tigzig.com](https://qrep-api.tigzig.com)

## Stack

- Python FastAPI + Uvicorn
- QuantStats 0.0.81 (portfolio analytics — 81 metrics per symbol)
- yfinance 0.2.66 (price data)
- fastapi-mcp (MCP server for AI tool integration)
- SlowAPI (rate limiting)
- ReportLab (PDF generation)

## Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/mcp/compare` | GET | None | MCP-optimized: 81 portfolio metrics per symbol (powered by QuantStats), no chart data (~6KB vs ~38KB). Up to 6 symbols. |
| `/mcp` | SSE | None | MCP SSE server — connect AI clients here for tool discovery |
| `/qpulse/compare` | POST | API key | Full comparison with metrics, time series, charts, price data |
| `/qpulse/analyze` | POST/GET | API key | Single-security tearsheet generation |
| `/qpulse/portfolio` | POST | API key | Portfolio comparison (weighted holdings) |
| `/qpulse/compare/export` | POST | API key | Export comparison as HTML/PDF |
| `/reports/{filename}` | GET | None | Serve generated report files |
| `/cleanup` | POST | Cleanup key | Cleanup old report files (cron-protected) |

## MCP Server

The `/mcp` SSE endpoint exposes a single tool (`mcp_compare_securities`) for AI clients. No API key required.

Connect your MCP client to: `https://qrep-api.tigzig.com/mcp`

The tool returns 81 portfolio metrics per symbol (powered by QuantStats), organized in these categories:
- **Returns & Performance** — Cumulative Return, CAGR, MTD/3M/6M/YTD/1Y, Best/Worst Day/Month/Year, Expected Daily/Monthly/Yearly
- **Risk-Adjusted Ratios** — Sharpe, Sortino, Omega, Calmar, Treynor, Information Ratio, Ulcer Performance Index
- **Drawdown** — Max Drawdown, Max DD Date/Period, Longest DD Days, Avg. Drawdown, Recovery Factor, Ulcer Index, Serenity Index
- **Volatility & Distribution** — Volatility (ann.), Skew, Kurtosis, R-squared
- **Benchmark-Relative** — Beta, Alpha, Correlation
- **Value-at-Risk & Tail Risk** — Daily VaR, Expected Shortfall (cVaR), Risk of Ruin, Kelly Criterion, Tail/Outlier Ratios
- **Win/Loss & Trade Statistics** — Win Days/Month/Quarter/Year %, Win/Loss Ratio, Profit Factor, Max Consecutive Wins/Losses, Gain/Pain Ratio

## Security Hardening

| # | Layer | What It Does |
|---|-------|-------------|
| 1 | API key enforcement | All `/qpulse/` endpoints require `X-API-Key` header — no key, no access |
| 2 | Fail-closed auth | If `QPULSE_API_KEY` env var is missing, all authenticated endpoints return 503 |
| 3 | Real client IP extraction | Custom `get_real_client_ip()` reads `X-Tigzig-User-IP` / `CF-Connecting-IP` / `X-Forwarded-For` — correct behind Cloudflare + Caddy reverse proxy |
| 4 | Per-IP rate limiting | Per-IP request rate limit via SlowAPI, configurable via `RATE_LIMIT` env var |
| 5 | Global rate limiting | Global request rate limit across all IPs via `shared_limit`, configurable via `GLOBAL_RATE_LIMIT` env var |
| 6 | Per-IP concurrency cap | Max simultaneous in-flight requests per IP, configurable via `MAX_CONCURRENT_PER_IP` env var |
| 7 | Global concurrency cap | Max simultaneous requests server-wide, configurable via `MAX_CONCURRENT_GLOBAL` env var |
| 8 | Split error codes | 503 for global capacity exceeded, 429 for per-IP rate/concurrency limits — no limit values disclosed |
| 9 | CORS credentials disabled | `allow_origins=["*"]` with `allow_credentials=False` — safe because auth uses API key headers, not cookies |
| 10 | Path traversal protection | `/reports/{filename}` uses `os.path.basename()` — blocks `../` attacks |
| 11 | Symbol format validation | Regex `^[A-Za-z0-9.\-=^$]{1,20}$` on all ticker inputs — blocks injection |
| 12 | Generic error messages | Internal errors return `"Analysis failed"` — no stack traces, file paths, or library names leaked |
| 13 | No version disclosure | Root endpoint returns only status — no library versions, config, or file paths |
| 14 | Pinned dependencies | All packages pinned to exact versions in requirements.txt |
| 15 | Automated report cleanup | `POST /cleanup` endpoint + daily cron job removes report files older than 3 days |
| 16 | Centralized API monitoring | All requests logged via `tigzig-api-monitor` middleware with request body and client IP capture |

Additional protections:
- **Concurrency counter leak protection** — `asyncio.shield` on counter release prevents permanent lockout from cancelled requests
- **Filename sanitization** — generated report filenames stripped of unsafe characters before writing to disk
- **All rate/concurrency values in env vars** — tunable per deployment without code changes

## Deployment

Docker container deployed via Coolify (Nixpacks) on Hetzner.

## Related

- **Frontend**: [qrep-security-analytics](https://github.com/amararun/qrep-security-analytics)
- **Live app**: [qrep.tigzig.com](https://qrep.tigzig.com)

## License

Apache License 2.0. See [LICENSE](LICENSE) and [THIRD-PARTY-LICENSES.md](THIRD-PARTY-LICENSES.md).

---

## Author

Built by [Amar Harolikar](https://www.linkedin.com/in/amarharolikar/)

Explore 30+ open source AI tools for analytics, databases & automation at [tigzig.com](https://tigzig.com)
