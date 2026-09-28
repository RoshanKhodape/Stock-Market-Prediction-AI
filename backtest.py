import yfinance as yf
import numpy as np
import pandas as pd


def backtest_strategy(
    symbol,
    initial_capital=100000,
    risk_per_trade=0.005,   # 0.5% risk per trade
    max_holding_days=7
):
    """
    AI Stock Predictor - Advanced Backtesting Engine

    Features:
    - SMA 20 / 50 / 200
    - RSI
    - ATR
    - ADX
    - Volume confirmation for stocks
    - Volume bypass for indexes
    - ATR Stop Loss
    - 1:2 Risk/Reward
    - Break-even after +1R
    - ATR trailing stop
    - Position sizing
    - Transaction cost
    - No overlapping trades
    - Debug information
    """

    try:

        print("\n")
        print("=" * 70)
        print("STARTING BACKTEST")
        print("SYMBOL:", symbol)
        print("=" * 70)

        # ==========================================================
        # INDEX DETECTION
        # ==========================================================

        is_index = symbol in [
            "^NSEI",
            "^NSEBANK",
            "^BSESN"
        ]

        if is_index:
            print("Instrument Type: INDEX")
        else:
            print("Instrument Type: STOCK")

        # ==========================================================
        # DOWNLOAD DATA
        # ==========================================================

        df = yf.download(
            symbol,
            period="5y",
            interval="1d",
            progress=False,
            auto_adjust=True
        )

        if df is None or df.empty:

            print("ERROR: No data downloaded.")

            return None

        # ==========================================================
        # FIX MULTIINDEX
        # ==========================================================

        if isinstance(df.columns, pd.MultiIndex):

            df.columns = df.columns.get_level_values(0)

        print("Downloaded rows:", len(df))

        # ==========================================================
        # REQUIRED COLUMNS
        # ==========================================================

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close"
        ]

        for col in required_columns:

            if col not in df.columns:

                print(
                    f"ERROR: Missing column {col}"
                )

                return None

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        # ==========================================================
        # VOLUME
        # ==========================================================

        if "Volume" in df.columns:

            df["Volume"] = pd.to_numeric(
                df["Volume"],
                errors="coerce"
            ).fillna(0)

        else:

            df["Volume"] = 0

        df = df.dropna(
            subset=required_columns
        ).copy()

        # ==========================================================
        # SMA 20
        # ==========================================================

        df["SMA_20"] = (
            df["Close"]
            .rolling(
                window=20,
                min_periods=20
            )
            .mean()
        )

        # ==========================================================
        # SMA 50
        # ==========================================================

        df["SMA_50"] = (
            df["Close"]
            .rolling(
                window=50,
                min_periods=50
            )
            .mean()
        )

        # ==========================================================
        # SMA 200
        # ==========================================================

        df["SMA_200"] = (
            df["Close"]
            .rolling(
                window=200,
                min_periods=200
            )
            .mean()
        )

        # ==========================================================
        # RSI
        # ==========================================================

        delta = df["Close"].diff()

        gain = delta.clip(
            lower=0
        )

        loss = -delta.clip(
            upper=0
        )

        avg_gain = (
            gain
            .rolling(
                window=14,
                min_periods=14
            )
            .mean()
        )

        avg_loss = (
            loss
            .rolling(
                window=14,
                min_periods=14
            )
            .mean()
        )

        rs = (
            avg_gain /
            avg_loss.replace(
                0,
                np.nan
            )
        )

        df["RSI"] = (
            100 -
            (
                100 /
                (1 + rs)
            )
        )

        # ==========================================================
        # ATR
        # ==========================================================

        previous_close = (
            df["Close"]
            .shift(1)
        )

        tr1 = (
            df["High"] -
            df["Low"]
        )

        tr2 = (
            df["High"] -
            previous_close
        ).abs()

        tr3 = (
            df["Low"] -
            previous_close
        ).abs()

        true_range = pd.concat(
            [
                tr1,
                tr2,
                tr3
            ],
            axis=1
        ).max(
            axis=1
        )

        df["ATR"] = (
            true_range
            .ewm(
                alpha=1 / 14,
                adjust=False
            )
            .mean()
        )

        # ==========================================================
        # ADX
        # ==========================================================

        up_move = (
            df["High"]
            .diff()
        )

        down_move = -(
            df["Low"]
            .diff()
        )

        plus_dm = np.where(
            (
                (up_move > down_move)
                &
                (up_move > 0)
            ),
            up_move,
            0
        )

        minus_dm = np.where(
            (
                (down_move > up_move)
                &
                (down_move > 0)
            ),
            down_move,
            0
        )

        plus_dm = pd.Series(
            plus_dm,
            index=df.index
        )

        minus_dm = pd.Series(
            minus_dm,
            index=df.index
        )

        atr_adx = (
            true_range
            .ewm(
                alpha=1 / 14,
                adjust=False
            )
            .mean()
        )

        plus_di = (
            100 *
            plus_dm
            .ewm(
                alpha=1 / 14,
                adjust=False
            )
            .mean()
            /
            atr_adx.replace(
                0,
                np.nan
            )
        )

        minus_di = (
            100 *
            minus_dm
            .ewm(
                alpha=1 / 14,
                adjust=False
            )
            .mean()
            /
            atr_adx.replace(
                0,
                np.nan
            )
        )

        di_sum = (
            plus_di +
            minus_di
        ).replace(
            0,
            np.nan
        )

        dx = (
            100 *
            (
                plus_di -
                minus_di
            ).abs()
            /
            di_sum
        )

        df["ADX"] = (
            dx
            .ewm(
                alpha=1 / 14,
                adjust=False
            )
            .mean()
        )

        # ==========================================================
        # VOLUME SMA
        # ==========================================================

        df["Volume_SMA_20"] = (
            df["Volume"]
            .rolling(
                window=20,
                min_periods=1
            )
            .mean()
        )

        # ==========================================================
        # REMOVE ONLY INDICATOR NaN
        # ==========================================================

        df = df.dropna(
            subset=[
                "SMA_20",
                "SMA_50",
                "SMA_200",
                "RSI",
                "ATR",
                "ADX"
            ]
        ).copy()

        print(
            "Rows after indicators:",
            len(df)
        )

        if df.empty:

            print(
                "ERROR: No rows remaining after indicators."
            )

            return None

        # ==========================================================
        # LATEST INDICATOR DEBUG
        # ==========================================================

        latest = df.iloc[-1]

        print("\n")
        print("LATEST MARKET DATA")
        print("-" * 50)

        print(
            "Close:",
            round(float(latest["Close"]), 2)
        )

        print(
            "SMA20:",
            round(float(latest["SMA_20"]), 2)
        )

        print(
            "SMA50:",
            round(float(latest["SMA_50"]), 2)
        )

        print(
            "SMA200:",
            round(float(latest["SMA_200"]), 2)
        )

        print(
            "RSI:",
            round(float(latest["RSI"]), 2)
        )

        print(
            "ATR:",
            round(float(latest["ATR"]), 2)
        )

        print(
            "ADX:",
            round(float(latest["ADX"]), 2)
        )

        print(
            "Volume:",
            round(float(latest["Volume"]), 2)
        )

        print(
            "Volume SMA:",
            round(
                float(
                    latest["Volume_SMA_20"]
                ),
                2
            )
        )

        # ==========================================================
        # CAPITAL / STATISTICS
        # ==========================================================

        capital = float(
            initial_capital
        )

        peak_equity = capital

        equity_curve = []

        drawdown_curve = []

        trades = []

        gross_profit = 0.0

        gross_loss = 0.0

        winning_trades = 0

        losing_trades = 0

        transaction_cost_rate = 0.0005

        # Debug counters
        buy_candidates = 0
        sell_candidates = 0

        trend_candidates = 0
        rsi_candidates = 0
        adx_candidates = 0
        volume_candidates = 0

        i = 0

        # ==========================================================
        # MAIN BACKTEST
        # ==========================================================

        while i < len(df) - 1:

            price = float(
                df["Close"].iloc[i]
            )

            sma20 = float(
                df["SMA_20"].iloc[i]
            )

            sma50 = float(
                df["SMA_50"].iloc[i]
            )

            sma200 = float(
                df["SMA_200"].iloc[i]
            )

            rsi = float(
                df["RSI"].iloc[i]
            )

            atr = float(
                df["ATR"].iloc[i]
            )

            adx = float(
                df["ADX"].iloc[i]
            )

            volume = float(
                df["Volume"].iloc[i]
            )

            avg_volume = float(
                df["Volume_SMA_20"].iloc[i]
            )

            # ======================================================
            # INDEX / STOCK FILTER
            # ======================================================

            if is_index:

                # Stricter trend filter for indexes.
                # ADX >= 20 avoids many weak/choppy signals.
                strong_trend = (
                    adx >= 20
                )

                volume_confirmed = True

            else:

                strong_trend = (
                    adx >= 20
                )

                volume_confirmed = (
                    avg_volume > 0
                    and
                    volume >=
                    avg_volume * 0.80
                )

            # ======================================================
            # DEBUG COUNTERS
            # ======================================================

            bullish_trend = (
                price > sma200
                and
                sma20 > sma50
                and
                price > sma20
            )

            bearish_trend = (
                price < sma200
                and
                sma20 < sma50
                and
                price < sma20
            )

            # Directional confirmation using DI.
            bullish_di = plus_di.iloc[i] > minus_di.iloc[i]
            bearish_di = minus_di.iloc[i] > plus_di.iloc[i]

            # Avoid entering at very weak/overextended RSI zones.
            buy_rsi = (
                rsi >= 55
                and
                rsi <= 68
            )

            sell_rsi = (
                rsi >= 32
                and
                rsi <= 45
            )

            if bullish_trend or bearish_trend:

                trend_candidates += 1

            if buy_rsi or sell_rsi:

                rsi_candidates += 1

            if strong_trend:

                adx_candidates += 1

            if volume_confirmed:

                volume_candidates += 1

            # ======================================================
            # SIGNAL
            # ======================================================

            signal = None

            # BUY
            if (
                bullish_trend
                and
                buy_rsi
                and
                bullish_di
                and
                strong_trend
                and
                volume_confirmed
            ):

                signal = "BUY"

                buy_candidates += 1

            # SELL
            elif (
                bearish_trend
                and
                sell_rsi
                and
                bearish_di
                and
                strong_trend
                and
                volume_confirmed
            ):

                signal = "SELL"

                sell_candidates += 1

            # ======================================================
            # FIRST SIGNAL DEBUG
            # ======================================================

            if i == 0:

                print("\n")
                print(
                    "SIGNAL FILTER DEBUG"
                )

                print("-" * 50)

                print(
                    "Symbol:",
                    symbol
                )

                print(
                    "Price:",
                    round(price, 2)
                )

                print(
                    "SMA20:",
                    round(sma20, 2)
                )

                print(
                    "SMA50:",
                    round(sma50, 2)
                )

                print(
                    "SMA200:",
                    round(sma200, 2)
                )

                print(
                    "RSI:",
                    round(rsi, 2)
                )

                print(
                    "ADX:",
                    round(adx, 2)
                )

                print(
                    "Strong Trend:",
                    strong_trend
                )

                print(
                    "Volume Confirmed:",
                    volume_confirmed
                )

                print(
                    "Bullish Trend:",
                    bullish_trend
                )

                print(
                    "Bearish Trend:",
                    bearish_trend
                )

                print(
                    "BUY RSI:",
                    buy_rsi
                )

                print(
                    "SELL RSI:",
                    sell_rsi
                )

                print(
                    "Signal:",
                    signal
                )

                print("-" * 50)

            # ======================================================
            # NO SIGNAL
            # ======================================================

            if signal is None:

                equity_curve.append(
                    capital
                )

                i += 1

                continue

            # ======================================================
            # NEXT DAY ENTRY
            # ======================================================

            entry_index = i + 1

            if entry_index >= len(df):

                break

            entry_price = float(
                df["Open"].iloc[
                    entry_index
                ]
            )

            # ======================================================
            # ATR RISK
            # ======================================================

            # 2 ATR stop, 2R target.
            # Combined with 0.5% account risk, this keeps individual
            # losses smaller without artificially changing trade outcomes.
            atr_multiplier = 2.0

            original_risk = (
                atr *
                atr_multiplier
            )

            if original_risk <= 0:

                i += 1

                continue

            # ======================================================
            # STOP LOSS / TARGET
            # ======================================================

            if signal == "BUY":

                stop_loss = (
                    entry_price -
                    original_risk
                )

                target = (
                    entry_price +
                    original_risk * 2
                )

            else:

                stop_loss = (
                    entry_price +
                    original_risk
                )

                target = (
                    entry_price -
                    original_risk * 2
                )

            # ==========================================================
            # POSITION SIZE
            # ==========================================================

            risk_amount = (
            capital *
            risk_per_trade
            )

            # ----------------------------------------------------------
            # INDEX
            # ----------------------------------------------------------
            # NIFTY / BANKNIFTY / SENSEX are index values.
            # We use fractional synthetic units so that each trade
# risks approximately 1% of current capital.
#
# This is NOT actual futures/options lot sizing.
# ----------------------------------------------------------

            if is_index:

                quantity = (
                    risk_amount /
                    original_risk
                )

                # Safety limits
                if quantity <= 0:

                    equity_curve.append(
                        capital
                    )

                    i += 1

                    continue

            # ----------------------------------------------------------
            # STOCK
            # ----------------------------------------------------------

            else:

                # Risk based quantity
                quantity_by_risk = int(
                    risk_amount /
                    original_risk
                )

            # Maximum 10% capital allocation
                max_trade_value = (
                    capital *
                    0.10
                )

                quantity_by_capital = int(
                    max_trade_value /
                    entry_price
                )

                quantity = min(
                    quantity_by_risk,
                    quantity_by_capital
                )

                # No valid stock quantity
                if quantity < 1:

                    equity_curve.append(
                        capital
                    )

                    i += 1

                    continue
                        

            # ======================================================
            # TRADE MANAGEMENT
            # ======================================================

            exit_price = None

            exit_reason = None

            actual_exit_index = min(
                entry_index +
                max_holding_days -
                1,
                len(df) - 1
            )

            highest_price = (
                entry_price
            )

            lowest_price = (
                entry_price
            )

            break_even_active = False

            # ======================================================
            # TRADE LOOP
            # ======================================================

            for j in range(
                entry_index,
                actual_exit_index + 1
            ):

                day_high = float(
                    df["High"].iloc[j]
                )

                day_low = float(
                    df["Low"].iloc[j]
                )

                # ==================================================
                # BUY
                # ==================================================

                if signal == "BUY":

                    highest_price = max(
                        highest_price,
                        day_high
                    )

                    # +1R = Break Even
                    if (
                        not break_even_active
                        and
                        highest_price >=
                        entry_price +
                        original_risk
                    ):

                        stop_loss = max(
                            stop_loss,
                            entry_price
                        )

                        break_even_active = True

                    # Trailing Stop
                    if break_even_active:

                        trailing_stop = (
                            highest_price -
                            atr * 1.5
                        )

                        stop_loss = max(
                            stop_loss,
                            trailing_stop
                        )

                    # Stop
                    if day_low <= stop_loss:

                        exit_price = (
                            stop_loss
                        )

                        if break_even_active:

                            exit_reason = (
                                "TRAIL_STOP"
                            )

                        else:

                            exit_reason = (
                                "STOP_LOSS"
                            )

                        actual_exit_index = j

                        break

                    # Target
                    if day_high >= target:

                        exit_price = target

                        exit_reason = (
                            "TARGET"
                        )

                        actual_exit_index = j

                        break

                # ==================================================
                # SELL
                # ==================================================

                else:

                    lowest_price = min(
                        lowest_price,
                        day_low
                    )

                    # +1R = Break Even
                    if (
                        not break_even_active
                        and
                        lowest_price <=
                        entry_price -
                        original_risk
                    ):

                        stop_loss = min(
                            stop_loss,
                            entry_price
                        )

                        break_even_active = True

                    # Trailing Stop
                    if break_even_active:

                        trailing_stop = (
                            lowest_price +
                            atr * 1.5
                        )

                        stop_loss = min(
                            stop_loss,
                            trailing_stop
                        )

                    # Stop
                    if day_high >= stop_loss:

                        exit_price = (
                            stop_loss
                        )

                        if break_even_active:

                            exit_reason = (
                                "TRAIL_STOP"
                            )

                        else:

                            exit_reason = (
                                "STOP_LOSS"
                            )

                        actual_exit_index = j

                        break

                    # Target
                    if day_low <= target:

                        exit_price = target

                        exit_reason = (
                            "TARGET"
                        )

                        actual_exit_index = j

                        break

            # ======================================================
            # TIME EXIT
            # ======================================================

            if exit_price is None:

                exit_price = float(
                    df["Close"].iloc[
                        actual_exit_index
                    ]
                )

                exit_reason = (
                    "TIME_EXIT"
                )

            # ======================================================
            # P&L
            # ======================================================

            if signal == "BUY":

                gross_pnl = (
                    exit_price -
                    entry_price
                ) * quantity

            else:

                gross_pnl = (
                    entry_price -
                    exit_price
                ) * quantity

            # ======================================================
            # TRANSACTION COST
            # ======================================================

            entry_value = (
                entry_price *
                quantity
            )

            exit_value = (
                exit_price *
                quantity
            )

            transaction_cost = (
                entry_value +
                exit_value
            ) * transaction_cost_rate

            net_pnl = (
                gross_pnl -
                transaction_cost
            )

            # ======================================================
            # CAPITAL UPDATE
            # ======================================================

            capital += net_pnl

            # ======================================================
            # WIN / LOSS
            # ======================================================

            if net_pnl > 0:

                winning_trades += 1

                gross_profit += net_pnl

            else:

                losing_trades += 1

                gross_loss += abs(
                    net_pnl
                )

            # ======================================================
            # SAVE TRADE
            # ======================================================

            trades.append({

                "signal":
                    signal,

                "entry":
                    round(
                        entry_price,
                        2
                    ),

                "exit":
                    round(
                        exit_price,
                        2
                    ),

                "stop_loss":
                    round(
                        stop_loss,
                        2
                    ),

                "target":
                    round(
                        target,
                        2
                    ),

                "quantity":
                    quantity,

                "profit":
                    round(
                        net_pnl,
                        2
                    ),

                "exit_reason":
                    exit_reason,

                "holding_days":
                    actual_exit_index -
                    entry_index +
                    1,

                "rsi":
                    round(
                        rsi,
                        2
                    ),

                "adx":
                    round(
                        adx,
                        2
                    ),

                "plus_di":
                    round(
                        float(plus_di.iloc[i]),
                        2
                    ),

                "minus_di":
                    round(
                        float(minus_di.iloc[i]),
                        2
                    )

            })

            # ======================================================
            # EQUITY
            # ======================================================

            equity_curve.append(
                capital
            )

            if capital > peak_equity:

                peak_equity = capital

            drawdown = (
                capital -
                peak_equity
            )

            drawdown_curve.append(
                drawdown
            )

            # ======================================================
            # PREVENT OVERLAPPING TRADES
            # ======================================================

            i = (
                actual_exit_index +
                1
            )

        # ==========================================================
        # FINAL STATISTICS
        # ==========================================================

        total_trades = len(
            trades
        )

        if total_trades > 0:

            win_rate = (
                winning_trades /
                total_trades
            ) * 100

        else:

            win_rate = 0

        total_profit = (
            capital -
            initial_capital
        )

        total_return = (
            total_profit /
            initial_capital
        ) * 100

        # ==========================================================
        # PROFIT FACTOR
        # ==========================================================

        if gross_loss > 0:

            profit_factor = (
                gross_profit /
                gross_loss
            )

        elif gross_profit > 0:

            profit_factor = 999.99

        else:

            profit_factor = 0

        # ==========================================================
        # MAX DRAWDOWN
        # ==========================================================

        max_drawdown = (
            min(drawdown_curve)
            if drawdown_curve
            else 0
        )

        # ==========================================================
        # SHARPE
        # ==========================================================

        if len(trades) > 1:

            returns = np.array(
                [
                    t["profit"] / initial_capital
                    for t in trades
                ],
                dtype=float
            )

            std_dev = np.std(
                returns
            )

            if std_dev > 0:

                # Trade-based Sharpe is only an approximation because
                # trades are not equally spaced in time.
                sharpe_ratio = (
                    np.mean(returns) /
                    std_dev
                ) * np.sqrt(len(returns))

            else:

                sharpe_ratio = 0

        else:

            sharpe_ratio = 0

        # ==========================================================
        # FINAL DEBUG SUMMARY
        # ==========================================================

        print("\n")
        print("=" * 70)
        print("BACKTEST SUMMARY")
        print("=" * 70)

        print(
            "Symbol:",
            symbol
        )

        print(
            "Total Trades:",
            total_trades
        )

        print(
            "BUY Candidates:",
            buy_candidates
        )

        print(
            "SELL Candidates:",
            sell_candidates
        )

        print(
            "Trend Candidates:",
            trend_candidates
        )

        print(
            "RSI Candidates:",
            rsi_candidates
        )

        print(
            "ADX Candidates:",
            adx_candidates
        )

        print(
            "Volume Candidates:",
            volume_candidates
        )

        print(
            "Winning Trades:",
            winning_trades
        )

        print(
            "Losing Trades:",
            losing_trades
        )

        print(
            "Win Rate:",
            round(
                win_rate,
                2
            ),
            "%"
        )

        print(
            "Net Profit:",
            round(
                total_profit,
                2
            )
        )

        print(
            "Profit Factor:",
            round(
                profit_factor,
                2
            )
        )

        print(
            "Max Drawdown:",
            round(
                max_drawdown,
                2
            )
        )

        print(
            "Sharpe:",
            round(
                sharpe_ratio,
                2
            )
        )

        print("=" * 70)
        print("BACKTEST COMPLETE")
        print("=" * 70)
        print("\n")

        # ==========================================================
        # RETURN
        # ==========================================================

        return {

            "total_trades":
                total_trades,

            "winning_trades":
                winning_trades,

            "losing_trades":
                losing_trades,

            "win_rate":
                round(
                    win_rate,
                    2
                ),

            "total_profit":
                round(
                    total_profit,
                    2
                ),

            "total_return":
                round(
                    total_return,
                    2
                ),

            "initial_capital":
                round(
                    initial_capital,
                    2
                ),

            "final_capital":
                round(
                    capital,
                    2
                ),

            "equity_curve":
                [
                    round(
                        x,
                        2
                    )
                    for x in equity_curve
                ],

            "drawdown_curve":
                [
                    round(
                        x,
                        2
                    )
                    for x in drawdown_curve
                ],

            "max_drawdown":
                round(
                    max_drawdown,
                    2
                ),

            "profit_factor":
                round(
                    profit_factor,
                    2
                ),

            "sharpe_ratio":
                round(
                    sharpe_ratio,
                    2
                ),

            "gross_profit":
                round(
                    gross_profit,
                    2
                ),

            "gross_loss":
                round(
                    gross_loss,
                    2
                ),

            "trades":
                trades
        }

    except Exception as e:

        print("\n")
        print("=" * 70)
        print("BACKTEST ERROR")
        print("=" * 70)
        print(
            "Symbol:",
            symbol
        )
        print(
            "Error:",
            str(e)
        )
        print("=" * 70)
        print("\n")

        return None