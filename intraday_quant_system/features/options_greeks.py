import math

def norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x: float) -> float:
    return math.exp(-x*x / 2.0) / math.sqrt(2.0 * math.pi)

class BlackScholesEngine:
    """
    Institutional-grade Black-Scholes engine for Indian F&O.
    Computes theoretical pricing and Option Greeks natively without heavy dependencies.
    """
    @staticmethod
    def calc_d1_d2(S, K, T, r, sigma):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2
        
    @staticmethod
    def call_price(S, K, T, r, sigma):
        if T <= 0: return max(0.0, S - K)
        d1, d2 = BlackScholesEngine.calc_d1_d2(S, K, T, r, sigma)
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
        
    @staticmethod
    def put_price(S, K, T, r, sigma):
        if T <= 0: return max(0.0, K - S)
        d1, d2 = BlackScholesEngine.calc_d1_d2(S, K, T, r, sigma)
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        
    @staticmethod
    def greeks(S, K, T, r, sigma, is_call=True):
        if T <= 0:
            return {'delta': 1.0 if is_call and S>K else (-1.0 if not is_call and K>S else 0.0), 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}
        d1, d2 = BlackScholesEngine.calc_d1_d2(S, K, T, r, sigma)
        
        gamma = norm_pdf(d1) / (S * sigma * math.sqrt(T))
        vega = S * norm_pdf(d1) * math.sqrt(T)
        
        if is_call:
            delta = norm_cdf(d1)
            theta = -(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm_cdf(d2)
        else:
            delta = norm_cdf(d1) - 1.0
            theta = -(S * norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm_cdf(-d2)
            
        return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega}
