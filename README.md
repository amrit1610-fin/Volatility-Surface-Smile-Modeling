Here is a complete, institutional-grade `README.md` file tailored to the exact architecture, challenges, and solutions you implemented in this project.

It highlights not just what the code does, but the quantitative reasoning behind *why* you built it this way—making it a perfect portfolio piece for Volatility Trading, Structuring, or Exotics Quant roles.

---

```markdown
# Volatility Surface & Smile Modeling (SVI + SABR + Heston)

An end-to-end quantitative pipeline that transforms raw, noisy option chain data into a smooth, arbitrage-free 3D volatility surface, and calibrates a dynamic stochastic volatility model to generate higher-order risk sensitivities.

## 📌 Business Context
A single flat implied volatility cannot reproduce the market-observed "smile" or "skew." Volatility trading desks and exotic structurers require:
1. **A Static Parametric Surface (SVI/SABR):** To mark vanilla option books consistently across strikes and tenors.
2. **A Dynamic Stochastic Model (Heston):** To accurately price path-dependent exotics and generate dynamic hedge ratios that move sensibly as spot and volatility shift.


---

## ⚙️ Pipeline Architecture

1. **Data Preprocessing & Filtering:** 
   * Ingests raw multi-tenor option chains.
   * Constructs an **OTM-Only Surface** (Out-of-the-Money Puts for $K < F$, OTM Calls for $K \ge F$) using Put-Call parity to eliminate illiquid ITM artifacts.
   * Applies strict liquidity constraints (e.g., max IV caps, $\pm 8\%$ log-moneyness windows, dropping data-starved tenors).
2. **Arbitrage Checking:** Scans for and corrects Butterfly (strike-space) and Calendar (time-space) arbitrage violations.
3. **Parametric Fitting (SVI & SABR):** Fits the Stochastic Volatility Inspired (SVI) and SABR models per expiry slice using SciPy's `differential_evolution` global optimizer, bound by the $b(1 + \vert{}\rho\vert{}) \le 2$ no-arbitrage constraint.
4. **Tenor Interpolation:** Uses Monotonic Cubic Interpolation (`PchipInterpolator`) across SVI parameters to construct a continuous, queryable 3D volatility surface without spline overshoot (Runge's phenomenon).
5. **Global Stochastic Calibration (Heston):** Calibrates the Heston parameters ($\kappa, \theta, \sigma, \rho, v_0$) to the idealized SVI surface using the Fourier-Cosine (COS) pricing method.
6. **Risk Ladder Generation:** Computes 1st and 2nd order Greeks (Delta, Vega, Vanna, Volga) via finite differences (bump-and-reprice).

---

## 📊 Key Quantitative Insights & Analysis

### 1. Global vs. Local Optimization in SVI
Initial attempts to fit the Raw SVI equation using local optimizers (SLSQP) resulted in flatlined curves due to getting trapped in local minima. Upgrading to a global optimizer (`differential_evolution`) successfully captured the true market smile, ensuring the base surface was accurate.

### 2. The "Frankenstein" Surface (Raw SVI Limitations)
Because Raw SVI fits slices completely independently, optimal parameters can jump wildly between adjacent tenors (e.g., trading high curvature $\sigma$ for a lower wing slope $b$). Interpolating between these disconnected parameter states creates artificial ridges in the 3D surface. 

### 3. Heston as a Structural Regularizer
When the Heston calibrator evaluates the interpolated SVI surface, it refuses to overfit to the artificial ridges caused by Raw SVI parameter jumps. Because the Heston model is governed by rigid, time-consistent stochastic calculus, it acts as a structural regularizer—smoothing out interpolation noise and outputting a mathematically flawless, arbitrage-free surface, albeit with a residual RMSE against the broken intermediate SVI grid.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Libraries:** 
  * `pandas`, `numpy`: Vectorized data manipulation and logical filtering.
  * `scipy.optimize` (`minimize`, `differential_evolution`): Global/Local calibration routines.
  * `scipy.interpolate` (`PchipInterpolator`): Monotonic surface interpolation.
  * `matplotlib`: 2D slice and 3D surface visualizations.

---

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/amrit1610-fin/Volatility-Surface-Smile-Modeling](https://github.com/amrit1610-fin/Volatility-Surface-Smile-Modeling)
   cd Volatility-Surface-Modeling

```

2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Run the pipeline:**
Replace your raw option chain CSVs in the `data/option_chains/` directory, update the spot/rate configurations in `main.py`, and execute:
```bash
python main.py

```



---

## 📂 Project Structure

```text
├── data/
│   ├── option_chains/                 # Raw CSV inputs
│   ├── data_cleaning.py               # Preprocessing & formatting
│   └── option_chain_cleaning.py       # Clean option chains individually
├── models/
│   ├── volatility_fitting.py          # SVI & SABR global optimization
│   ├── heston_calibration.py          # Heston COS Pricer & Calibrator
├── utils/
│   ├── arbitrage_checks.py            # Butterfly/Calendar violation logic
│   ├── implied_vol.py                 # Universal Black-Scholes Brentq solver
│   └── tenor_interpolation.py         # Pchip SVI parameter interpolation
│   └── greeks.py                      # Finite difference Vanna/Volga engine
├── main.py                            # Main execution pipeline
└── README.md

```

---

## 🔮 Future Enhancements

* **Surface SVI (SSVI):** Upgrade the parametric base layer from Raw SVI to SSVI to guarantee the absence of calendar arbitrage and ensure smooth parameter evolution across tenors, which will drive the Heston calibration RMSE to near-zero.
* **Bid-Ask Corridor Penalty:** Replace the mid-price Sum of Squared Errors (SSE) objective function with a bid-ask corridor penalty to natively weight calibration toward highly liquid options.

```

```