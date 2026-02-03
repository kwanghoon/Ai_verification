import math
from z3 import Real, Bool, If, And, Or, Sum, Solver, sat, unsat, Not

# ----------------------------
# Original network (for numeric re-check)
# ----------------------------
def sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def net_logit_float(x1: float, x2: float) -> float:
    t1 = 3.4243*x1 + 3.4299*x2 - 5.3119
    t2 = 4.4863*x1 + 4.4830*x2 - 1.7982
    h1 = sig(t1)
    h2 = sig(t2)
    z  = -7.1722*h1 + 6.7997*h2 - 3.0611
    return z

def xor_label(x0):
    # XOR on {0,1}^2 : 1 iff exactly one input is 1
    return int((x0[0] + x0[1]) % 2 == 1)

# ----------------------------
# Helpers: linear range of affine form a*x + b*y + c over box
# ----------------------------
def affine_range_over_box(a, b, c, x1_lo, x1_hi, x2_lo, x2_hi):
    # since it's linear, min/max at corners
    vals = [
        a*x1_lo + b*x2_lo + c,
        a*x1_lo + b*x2_hi + c,
        a*x1_hi + b*x2_lo + c,
        a*x1_hi + b*x2_hi + c,
    ]
    return (min(vals), max(vals))

# ----------------------------
# Sound PWL outer-approx for sigmoid on [L,U], with 0 included as breakpoint
# Uses segment selection booleans + Big-M gating.
# On each segment [a,b] that is entirely <=0 (convex) or >=0 (concave):
#   convex (t<=0): chord is UPPER, tangents at endpoints are LOWER
#   concave(t>=0): chord is LOWER, tangents at endpoints are UPPER
# ----------------------------
def sigmoid_outer_pwl(slv: Solver, t, h, L, U, K=32, name="s", M=100.0):
    assert L <= U
    # build breakpoints including 0 if inside
    breaks = [L + (U-L)*i/K for i in range(K+1)]
    if L < 0.0 < U:
        breaks.append(0.0)
    breaks = sorted(set([round(b, 12) for b in breaks]))  # stabilize duplicates

    segs = []
    cons = []
    cons += [t >= L, t <= U, h >= 0.0, h <= 1.0]

    for i in range(len(breaks)-1):
        a = breaks[i]
        b = breaks[i+1]
        if b - a < 1e-12:
            continue
        # ensure segment doesn't cross 0 (shouldn't if 0 included)
        if a < 0.0 < b:
            # split should have prevented this; skip defensively
            continue

        s = Bool(f"{name}_seg_{i}")
        segs.append((s, a, b))

    # exactly one segment active
    cons += [Sum([If(s, 1, 0) for (s,_,_) in segs]) == 1]

    # precompute lines (as float constants) then embed in z3 reals
    for (s, a, b) in segs:
        sa = sig(a)
        sb = sig(b)

        # chord line through (a,sa), (b,sb): y = m*t + c
        m_ch = (sb - sa)/(b - a)
        c_ch = sa - m_ch*a

        # tangent at u: y = sig(u) + sig'(u)*(t-u)
        # sig'(u) = sig(u)*(1-sig(u))
        def tan_params(u):
            su = sig(u)
            d  = su*(1.0-su)
            m  = d
            c  = su - d*u
            return m, c

        m_ta, c_ta = tan_params(a)
        m_tb, c_tb = tan_params(b)

        # segment activation: t in [a,b]
        cons += [
            Or(Not(s), t >= a),
            Or(Not(s), t <= b),
        ]

        # Determine convex/concave by sign region
        if b <= 0.0:
            # convex: tangents are LOWER, chord is UPPER
            # lower: h >= tan(a), h >= tan(b)
            # upper: h <= chord
            cons += [
                Or(Not(s), h >= m_ta*t + c_ta),
                Or(Not(s), h >= m_tb*t + c_tb),
                Or(Not(s), h <= m_ch*t + c_ch),
            ]
        elif a >= 0.0:
            # concave: chord is LOWER, tangents are UPPER
            cons += [
                Or(Not(s), h <= m_ta*t + c_ta),
                Or(Not(s), h <= m_tb*t + c_tb),
                Or(Not(s), h >= m_ch*t + c_ch),
            ]
        else:
            # should not happen if 0 included and no cross
            # fallback: very loose safe bounds
            cons += [True]

    slv.add(cons)

# ----------------------------
# Build and check for a counterexample (existential)
# Return:
#   "unsat" => robust proven (sound)
#   "sat" => candidate from outer-approx; we re-check on real net
# ----------------------------
def check_one_x0(eps=0.1, x0=(0,0), K=32, M=100.0, verbose=True):
    x1_lo = max(0.0, x0[0]-eps); x1_hi = min(1.0, x0[0]+eps)
    x2_lo = max(0.0, x0[1]-eps); x2_hi = min(1.0, x0[1]+eps)

    slv = Solver()

    x1 = Real("x1")
    x2 = Real("x2")
    slv.add(x1 >= x1_lo, x1 <= x1_hi, x2 >= x2_lo, x2 <= x2_hi)

    # hidden pre-activations
    t1 = Real("t1")
    t2 = Real("t2")
    slv.add(t1 == 3.4243*x1 + 3.4299*x2 - 5.3119)
    slv.add(t2 == 4.4863*x1 + 4.4830*x2 - 1.7982)

    # tighten ranges for PWL
    L1,U1 = affine_range_over_box(3.4243, 3.4299, -5.3119, x1_lo, x1_hi, x2_lo, x2_hi)
    L2,U2 = affine_range_over_box(4.4863, 4.4830, -1.7982, x1_lo, x1_hi, x2_lo, x2_hi)

    h1 = Real("h1")
    h2 = Real("h2")

    sigmoid_outer_pwl(slv, t1, h1, L1, U1, K=K, name="h1", M=M)
    sigmoid_outer_pwl(slv, t2, h2, L2, U2, K=K, name="h2", M=M)

    # output logit
    z = Real("z")
    slv.add(z == -7.1722*h1 + 6.7997*h2 - 3.0611)

    y0 = xor_label(x0)
    # misclassification condition (threshold 0.5 <=> z sign)
    if y0 == 1:
        slv.add(z <= 0)
    else:
        slv.add(z >= 0)

    res = slv.check()
    if res == unsat:
        if verbose:
            print(f"x0={x0}, eps={eps}: UNSAT (sound => robust proven), K={K}")
        return "unsat", None

    if res != sat:
        if verbose:
            print(f"x0={x0}, eps={eps}: {res} (unknown-ish), K={K}")
        return str(res), None

    m = slv.model()
    # Extract a candidate point
    def to_float(z3val):
        s = str(z3val)
        if "/" in s:
            p,q = s.split("/")
            return float(p)/float(q)
        return float(s)

    x1v = to_float(m[x1])
    x2v = to_float(m[x2])

    # Re-check on the TRUE network
    z_true = net_logit_float(x1v, x2v)
    mis_true = (z_true <= 0.0) if (y0==1) else (z_true >= 0.0)

    if verbose:
        print(f"x0={x0}, eps={eps}: SAT in outer-approx (candidate)")
        print(f"  candidate x=({x1v:.6f},{x2v:.6f}), true z={z_true:.6f}, true_mis={mis_true}")

    if mis_true:
        return "sat_true_counterexample", (x1v, x2v, z_true)
    else:
        return "sat_spurious", (x1v, x2v, z_true)

def main():
    eps = 0.1
    K = 32   # increase to 64 if many spurious SAT
    M = 100.0
    for x0 in [(0,0),(0,1),(1,0),(1,1)]:
        status, info = check_one_x0(eps=eps, x0=x0, K=K, M=M, verbose=True)
        if status == "sat_spurious":
            print("  -> Spurious due to loose PWL. Increase K (e.g., 64) or tighten ranges.")
        elif status == "sat_true_counterexample":
            print("  -> REAL counterexample found.")
        print("-"*60)

if __name__ == "__main__":
    main()
