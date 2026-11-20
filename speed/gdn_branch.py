"""Output-only short GDN continuations from a read-only FP32 prefix bank (SM120)."""

import torch
import triton
import triton.language as tl

from triton.language.extra.cuda import libdevice

from speed.lm_prefill import gated_norm


@triton.jit
def _branch_conv(
    X,
    Tail,
    Weight,
    Bias,
    Off,
    Images,
    Y,
    AB,
    ALog,
    BiasDt,
    Beta,
    G,
    SAB: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    SX: tl.constexpr,
    SB: tl.constexpr,
    SC: tl.constexpr,
    ST: tl.constexpr,
    UNIFORM: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TAPS: tl.constexpr,
    BT: tl.constexpr,
    B2=None,
    G2=None,
    T: tl.constexpr = 0,
    M: tl.constexpr = 0,
):
    b = tl.program_id(0)
    if UNIFORM:
        start, length, img = b * UNIFORM, UNIFORM, b
    else:
        start = tl.load(Off + b)
        length = tl.load(Off + b + 1) - start
        img = tl.load(Images + b)
    t = tl.program_id(1) * BT + tl.arange(0, BT)
    c = tl.program_id(2) * 128 + tl.arange(0, 128)
    acc = tl.full((BT, 128), 0, tl.float32)
    for i in tl.static_range(TAPS):
        pos = t + i + 1 - TAPS
        cur = tl.load(X + (start + pos[:, None]) * SX + c[None, :], ((pos >= 0) & (t < length))[:, None], 0)
        prev = tl.load(
            Tail + img * SB + c[None, :] * SC + (pos[:, None] + TAPS - 1) * ST, ((pos < 0) & (t < length))[:, None], 0
        )
        w = tl.load(Weight + c * TAPS + i).to(tl.float32)
        acc = tl.fma((cur + prev).to(tl.float32), w[None, :], acc)
    if HAS_BIAS:
        acc += tl.load(Bias + c)[None, :]
    x = acc.to(tl.bfloat16).to(tl.float32)
    y = (x * tl.sigmoid(x)).to(tl.bfloat16)
    if T:
        row = b * T + t
        if tl.program_id(2) < 2 * KD // 128:
            yf = y.to(tl.float32)
            y = (yf * (1.0 / tl.sqrt(tl.sum(yf * yf, 1) + 1.0e-6))[:, None]).to(tl.bfloat16)
            base = Y + row[:, None] * KD + c[None, :]
            if tl.program_id(2) >= KD // 128:
                base += (M * T - 1) * KD
        else:
            base = Y + M * T * 2 * KD + row[:, None] * VD + c[None, :] - 2 * KD
            h = tl.program_id(2) - 2 * KD // 128
            src = (start + t) * (VD // 128) + h
            dst = row * (VD // 128) + h
            tl.store(B2 + dst, tl.load(Beta + src, t < length, 0), t < T)
            tl.store(G2 + dst, tl.load(G + src, t < length, 0), t < T)
        tl.store(base, y, (t < T)[:, None])
    else:
        tl.store(Y + (start + t[:, None]) * (2 * KD + VD) + c[None, :], y, (t < length)[:, None])
    if SAB and tl.program_id(2) >= 2 * KD // 128:
        h = tl.program_id(2) - 2 * KD // 128
        hv = VD // 128
        ab = AB + (start + t) * SAB + h
        bval = tl.load(ab, t < length, 0).to(tl.float32)
        aval = tl.load(ab + hv, t < length, 0).to(tl.float32) + tl.load(BiasDt + h).to(tl.float32)
        softplus = tl.maximum(aval, 0.0) + libdevice.log1p(tl.exp(-tl.abs(aval)))
        tl.store(Beta + (start + t) * hv + h, tl.sigmoid(bval), t < length)
        tl.store(G + (start + t) * hv + h, -tl.exp(tl.load(ALog + h).to(tl.float32)) * softplus, t < length)


@triton.jit
def _prod(a, b):
    # The correction products retain FP32 accuracy through the triangular solve.
    ah, bh = a.to(tl.bfloat16), b.to(tl.bfloat16)
    al = (a - ah.to(tl.float32)).to(tl.bfloat16)
    bl = (b - bh.to(tl.float32)).to(tl.bfloat16)
    return tl.dot(ah, bh) + (tl.dot(al, bh) + tl.dot(ah, bl))


@triton.jit
def _group(block, Images, GroupOff, SHARED: tl.constexpr, UNIFORM: tl.constexpr, INDEXED: tl.constexpr):
    if INDEXED:
        img, first, last = block, tl.load(GroupOff + block), tl.load(GroupOff + block + 1)
    else:
        first, last = block * SHARED, (block + 1) * SHARED
        if SHARED > 1:
            img = block * 0
        elif UNIFORM:
            img = block
        else:
            img = tl.load(Images + block)
    return img.to(tl.int64), first.to(tl.int32), last.to(tl.int32)


@triton.jit
def _split_state(state):
    hi = state.to(tl.bfloat16)
    rem = state - hi.to(tl.float32)
    mid = rem.to(tl.bfloat16)
    return hi, mid, (rem - mid.to(tl.float32)).to(tl.bfloat16)


@triton.jit
def _state_dot(x, hi, mid, lo):
    return tl.dot(x, hi) + (tl.dot(x, mid) + tl.dot(x, lo))


@triton.jit
def _seed_products(
    X,
    S,
    P,
    Off,
    Blocks: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    SH: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    BV: tl.constexpr,
):
    h = tl.program_id(0)
    kh = h // (HV // H)
    d = tl.arange(0, 128)
    v = tl.program_id(1) * BV + tl.arange(0, BV)
    hi, mid, lo = _split_state(tl.load(S + h * SH + d[:, None] * SK + v[None, :] * SV))
    n = tl.load(Off + Blocks)
    for start in range(0, n, 32):
        row = start + tl.arange(0, 32)
        src = X + row[:, None] * ((2 * H + HV) * 128) + kh * 128 + d[None, :]
        q = tl.load(src, (row < n)[:, None], 0)
        k = tl.load(src + H * 128, (row < n)[:, None], 0)
        dst = P + (row * 2 * HV + h)[:, None] * 128 + v[None, :]
        tl.store(dst, _state_dot(q, hi, mid, lo), (row < n)[:, None])
        tl.store(dst + HV * 128, _state_dot(k, hi, mid, lo), (row < n)[:, None])


@triton.jit
def _branch_wy(
    X,
    Beta,
    G,
    S,
    Seed,
    Off,
    Images,
    Order,
    GroupOff,
    O,
    STAGED: tl.constexpr,
    INDEXED: tl.constexpr,
    LONG: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SS2: tl.constexpr,
    SS3: tl.constexpr,
    UNIFORM: tl.constexpr,
    SHARED: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    STEPS: tl.constexpr,
):
    """Solve for the rank-one innovations by doubling a strictly lower triangular matrix."""
    block, h = tl.program_id(0), tl.program_id(1)
    kv = h // (HV // H)
    v = tl.program_id(2) * BV + tl.arange(0, BV)
    d = tl.arange(0, 128)
    t = tl.arange(0, BT)
    img, first, last = _group(block, Images, GroupOff, SHARED, UNIFORM, INDEXED)
    if STAGED:
        state = tl.full((128, BV), 0, tl.float32)
    else:
        state = tl.load(S + img * SS0 + h * SS1 + d[:, None] * SS2 + v[None, :] * SS3)
    seed_hi, seed_mid, seed_lo = _split_state(state)
    for pos in range(first, last):
        b = tl.load(Order + pos) if INDEXED else pos
        if UNIFORM:
            start, length = b * UNIFORM, UNIFORM
        else:
            start = tl.load(Off + b)
            length = tl.load(Off + b + 1) - start
        carry = 1.0
        live = state
        hi, mid, lo = seed_hi, seed_mid, seed_lo
        chunks = tl.cdiv(length, BT) if LONG else 1
        for chunk in range(chunks):
            row = start + chunk * BT + t
            ptr = X + row[:, None] * ((2 * H + HV) * 128)
            q = tl.load(ptr + kv * 128 + d[None, :], (chunk * BT + t < length)[:, None], 0)
            k = tl.load(ptr + (H + kv) * 128 + d[None, :], (chunk * BT + t < length)[:, None], 0)
            val = tl.load(ptr + (2 * H + h) * 128 + v[None, :], (chunk * BT + t < length)[:, None], 0).to(tl.float32)
            beta = tl.load(Beta + row * HV + h, chunk * BT + t < length, 0).to(tl.float32)
            g = tl.load(G + row * HV + h, chunk * BT + t < length, 0).to(tl.float32)
            gamma = tl.cumsum(g, 0)
            decay = tl.exp(tl.minimum(gamma[:, None] - gamma[None, :], 0.0))
            kf, qf = k.to(tl.float32), q.to(tl.float32)
            kn = tl.rsqrt(tl.sum(kf * kf, 1) + 1.0e-6)
            qn = tl.rsqrt(tl.sum(qf * qf, 1) + 1.0e-6) * (128.0**-0.5)
            kk = tl.dot(k, tl.trans(k)) * kn[:, None] * kn[None, :]
            qk = tl.dot(q, tl.trans(k)) * qn[:, None] * kn[None, :]
            if STAGED:
                addr = Seed + (row * 2 * HV + h)[:, None] * 128 + v[None, :]
                qs = tl.load(addr, (chunk * BT + t < length)[:, None], 0)
                ks = tl.load(addr + HV * 128, (chunk * BT + t < length)[:, None], 0)
                if LONG and chunk > 0:
                    qs = carry * qs + _state_dot(q, hi, mid, lo)
                    ks = carry * ks + _state_dot(k, hi, mid, lo)
            else:
                qs = _state_dot(q, hi, mid, lo)
                ks = _state_dot(k, hi, mid, lo)
            prefix = tl.exp(gamma)
            delta = beta[:, None] * (val - (prefix * kn)[:, None] * ks)
            # (I-X)^-1 = (I+X)(I+X^2)... because X^BT is zero.
            x = tl.where(t[:, None] > t[None, :], -beta[:, None] * kk * decay, 0.0)
            for step in tl.static_range(STEPS):
                delta += _prod(x, delta)
                if step + 1 < STEPS:
                    x = _prod(x, x)
            a = tl.where(t[:, None] >= t[None, :], qk * decay, 0.0)
            y = (prefix * qn)[:, None] * qs + _prod(a, delta)
            tl.store(O + (row * HV + h)[:, None] * 128 + v[None, :], y, (chunk * BT + t < length)[:, None])
            if LONG and chunk + 1 < chunks:
                end_gate = tl.min(gamma, 0)
                carry *= tl.exp(end_gate)
                gain = kn * tl.exp(end_gate - gamma)
                live = live * tl.exp(end_gate) + _prod(tl.trans(k).to(tl.float32), gain[:, None] * delta)
                hi, mid, lo = _split_state(live)


@triton.jit
def _branch_recurrent(
    X,
    Beta,
    G,
    S,
    Off,
    Images,
    Order,
    GroupOff,
    O,
    INDEXED: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SS2: tl.constexpr,
    SS3: tl.constexpr,
    UNIFORM: tl.constexpr,
    SHARED: tl.constexpr,
    BV: tl.constexpr,
):
    block, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, 128)
    v = tl.program_id(2) * BV + tl.arange(0, BV)
    img, first, last = _group(block, Images, GroupOff, SHARED, UNIFORM, INDEXED)
    prefix = tl.load(S + img * SS0 + h * SS1 + d[:, None] * SS2 + v[None, :] * SS3)
    for pos in range(first, last):
        b = tl.load(Order + pos) if INDEXED else pos
        if UNIFORM:
            start, length = b * UNIFORM, UNIFORM
        else:
            start = tl.load(Off + b)
            length = tl.load(Off + b + 1) - start
        state = prefix
        for i in range(length):
            row = start + i
            base = X + row * ((2 * H + HV) * 128)
            kh = h // (HV // H)
            q = tl.load(base + kh * 128 + d).to(tl.float32)
            k = tl.load(base + (H + kh) * 128 + d).to(tl.float32)
            value = tl.load(base + (2 * H + h) * 128 + v).to(tl.float32)
            q = q / tl.sqrt(tl.sum(q * q, 0) + 1.0e-6) * (128.0**-0.5)
            k = k / tl.sqrt(tl.sum(k * k, 0) + 1.0e-6)
            state *= tl.exp(tl.load(G + row * HV + h))
            delta = (value - tl.sum(state * k[:, None], 0)) * tl.load(Beta + row * HV + h).to(tl.float32)
            state += k[:, None] * delta[None, :]
            tl.store(O + (row * HV + h) * 128 + v, tl.sum(state * q[:, None], 0))


def branch_core(
    m,
    mixed,
    beta,
    g,
    state,
    tail,
    blocks,
    max_length,
    offsets=None,
    images=None,
    recurrent=False,
    raw_ab=None,
    groups=(None, None),
    staged=None,
):
    """Packed right-padded blocks, either all one length in bank order or indexed by offsets."""
    n = len(mixed)
    kd, vd = m.key_dim, m.value_dim
    conv = mixed.new_empty((n, 2 * kd + vd))
    uniform = max_length if offsets is None else 0
    w, bias = m.conv1d.weight, m.conv1d.bias
    if raw_ab is not None:
        beta = mixed.new_empty((n, vd // 128))
        g = torch.empty_like(beta, dtype=torch.float32)
    _branch_conv[(blocks, triton.cdiv(max_length, 16), (2 * kd + vd) // 128)](
        mixed,
        tail,
        w,
        bias,
        offsets,
        images,
        conv,
        raw_ab,
        m.A_log,
        m.dt_bias,
        beta,
        g,
        raw_ab.stride(0) if raw_ab is not None else 0,
        kd,
        vd,
        mixed.stride(0),
        *tail.stride(),
        uniform,
        bias is not None,
        w.shape[-1],
        16,
    )
    # Padded projection rows have no block and are mapped to zeros by the reference gather.
    out = mixed.new_zeros((n, vd // 128, 128))
    bt = min(32, max(16, triton.next_power_of_2(max_length)))
    shared = blocks if state.shape[0] == 1 else 1
    order, group_offsets = groups
    indexed = group_offsets is not None
    nw = state.shape[0] if indexed else blocks // shared
    if recurrent:
        _branch_recurrent[(nw, vd // 128, 8)](
            conv,
            beta,
            g,
            state,
            offsets,
            images,
            order,
            group_offsets,
            out,
            indexed,
            kd // 128,
            vd // 128,
            *state.stride(),
            uniform,
            shared,
            16,
        )
        return out
    if staged is None:
        staged = shared >= 8 and max_length > 32
    seed = None
    if staged:
        seed = torch.empty((n, 2, vd), device=mixed.device, dtype=torch.float32)
        _seed_products[(vd // 128, 8)](
            conv, state, seed, offsets, blocks, kd // 128, vd // 128, *state.stride()[1:], 16, num_stages=1
        )
        shared, nw = 1, blocks
    _branch_wy[(nw, vd // 128, 128 // 32)](
        conv,
        beta,
        g,
        state,
        seed,
        offsets,
        images,
        order,
        group_offsets,
        out,
        staged,
        indexed,
        max_length > bt,
        kd // 128,
        vd // 128,
        *state.stride(),
        uniform,
        shared,
        bt,
        32,
        bt.bit_length() - 1,
        num_stages=1,
    )
    return out


def branch_output(
    m,
    mixed,
    z,
    beta,
    g,
    state,
    tail,
    blocks,
    max_length,
    offsets=None,
    images=None,
    recurrent=False,
    raw_ab=None,
    groups=(None, None),
    staged=None,
):
    out = branch_core(
        m, mixed, beta, g, state, tail, blocks, max_length, offsets, images, recurrent, raw_ab, groups, staged
    )
    return gated_norm(m, out, z).view(len(mixed), m.value_dim)


@triton.jit
def _chunk_output(
    Q,
    K,
    W,
    U,
    G,
    S,
    Off,
    Images,
    Order,
    GroupOff,
    O,
    H: tl.constexpr,
    HV: tl.constexpr,
    T: tl.constexpr,
    SHARED: tl.constexpr,
    INDEXED: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    block, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, 128)
    v = tl.program_id(2) * BV + tl.arange(0, BV)
    t = tl.arange(0, BT)
    img, first, last = _group(block, Images, GroupOff, SHARED, 0, INDEXED)
    prefix = tl.load(S + img * S0 + h * S1 + d[:, None] * S2 + v[None, :] * S3)
    for slot in range(first, last):
        b = tl.load(Order + slot) if INDEXED else slot
        start = tl.load(Off + b)
        length = tl.load(Off + b + 1) - start
        state = prefix
        for c in range(tl.cdiv(length, BT)):
            i = c * BT + t
            r = b * T + i
            good = i < length
            dk = ((r * H + h // (HV // H)) * 128)[:, None] + d[None, :]
            q = tl.load(Q + dk, good[:, None], 0)
            k = tl.load(K + dk, good[:, None], 0)
            w = tl.load(W + ((r * HV + h) * 128)[:, None] + d[None, :], good[:, None], 0)
            u = tl.load(U + ((r * HV + h) * 128)[:, None] + v[None, :], good[:, None], 0)
            g = tl.load(G + r * HV + h, good, 0)
            # FLA's chunk path rounds the boundary state before both contractions
            # and rounds the new values before the within-chunk output product.
            hbf = state.to(tl.bfloat16)
            delta = u.to(tl.float32) - tl.dot(w, hbf)
            scores = tl.dot(q, tl.trans(k))
            scores = tl.where(t[:, None] >= t[None, :], scores * tl.exp2(tl.minimum(g[:, None] - g[None, :], 0.0)), 0.0)
            seed = tl.dot(q, hbf) * tl.exp2(g)[:, None]
            y = seed * (128.0**-0.5) + tl.dot(scores.to(tl.bfloat16), delta.to(tl.bfloat16)) * (128.0**-0.5)
            tl.store(O + (((start + i) * HV + h) * 128)[:, None] + v[None, :], y, good[:, None])
            if T > 64 and (c + 1) * BT < length:
                end_g = tl.min(g, 0)
                dv = (delta * tl.exp2(end_g - g)[:, None]).to(tl.bfloat16)
                state = tl.dot(tl.trans(k), dv, state * tl.exp2(end_g))


def load_chunk_branch():
    try:
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd_intra, chunk_local_cumsum
    except ImportError:
        return None

    def chunk_branch(m, mixed, z, beta, g, state, tail, blocks, length, offsets, images, groups):
        kd, vd = m.key_dim, m.value_dim
        width = blocks * length
        buf = mixed.new_empty(width * (2 * kd + vd))
        b2 = beta.new_empty((blocks, length, vd // 128))
        g2 = g.new_empty(b2.shape)
        w, bias = m.conv1d.weight, m.conv1d.bias
        _branch_conv[(blocks, triton.cdiv(length, 16), (2 * kd + vd) // 128)](
            mixed,
            tail,
            w,
            bias,
            offsets,
            images,
            buf,
            None,
            None,
            None,
            beta,
            g,
            0,
            kd,
            vd,
            mixed.stride(0),
            *tail.stride(),
            0,
            bias is not None,
            w.shape[-1],
            16,
            b2,
            g2,
            length,
            blocks,
        )
        q, k, v = (x.view(blocks, length, -1, 128) for x in buf.split([width * kd, width * kd, width * vd]))
        log2g = chunk_local_cumsum(g2, chunk_size=64, scale=1.4426950408889634)
        w, u, _ = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=log2g, beta=b2, chunk_size=64)
        out = mixed.new_zeros((len(mixed), vd // 128, 128))
        order, group_offsets = groups
        shared = blocks if state.shape[0] == 1 else 1
        nw = state.shape[0] if group_offsets is not None else blocks // shared
        bt = min(64, max(16, triton.next_power_of_2(length)))
        _chunk_output[(nw, vd // 128, 8)](
            q,
            k,
            w,
            u,
            log2g,
            state,
            offsets,
            images,
            order,
            group_offsets,
            out,
            kd // 128,
            vd // 128,
            length,
            shared,
            group_offsets is not None,
            *state.stride(),
            bt,
            16,
            num_stages=1,
        )
        return gated_norm(m, out, z).view(len(mixed), vd)

    return chunk_branch


@triton.jit
def _branch_attention(
    Q,
    K,
    V,
    Gate,
    Pk,
    Pv,
    Valid,
    Tokens,
    Off,
    Images,
    O,
    L: tl.constexpr,
    T: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    VS: tl.constexpr,
    GS: tl.constexpr,
    GH: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    V0: tl.constexpr,
    V1: tl.constexpr,
    V2: tl.constexpr,
    MASK_STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
    UNIFORM: tl.constexpr,
    CAUSAL: tl.constexpr,
    PACKED_KV: tl.constexpr,
    BANK: tl.constexpr,
    BQ: tl.constexpr,
    BN: tl.constexpr,
):
    b, h = tl.program_id(0), tl.program_id(1)
    if UNIFORM:
        start, length, img = b * UNIFORM, UNIFORM, b
    else:
        start = tl.load(Off + b)
        length = tl.load(Off + b + 1) - start
        img = tl.load(Images + b)
    i = tl.program_id(2) * BQ + tl.arange(0, BQ)
    n = tl.arange(0, BN)
    d = tl.arange(0, 256)
    kh = h // (HQ // HK)
    q = tl.load(Q + ((start + i) * HQ + h)[:, None] * 256 + d[None, :], (i < length)[:, None], 0)
    kp = Pk + img * K0 + kh * K1
    vp = Pv + img * V0 + kh * V1
    # Find one maximum over both cache segments. Bank math rounds the unscaled
    # score and each separate value product to BF16, as in the two eager BMMs.
    mx = tl.full((BQ,), -float("inf"), tl.float32)
    for part in tl.static_range(2):
        size = L if part == 0 else T
        for off in range(0, size, BN):
            j = off + n
            if part == 0:
                slot = tl.load(Tokens + img * L + j, j < L, 0) if PACKED_KV else j
                k = tl.load(kp + slot[:, None] * K2 + d[None, :], (j < L)[:, None], 0)
                valid = tl.load(Valid + img * MASK_STRIDE + j, j < L, 0)
            else:
                k = tl.load(K + ((start + j) * HK + kh)[:, None] * 256 + d[None, :], (j < length)[:, None], 0)
                valid = j < length
            score = tl.dot(q, tl.trans(k))
            if BANK:
                score = score.to(tl.bfloat16).to(tl.float32)
            score = tl.where(valid[None, :], score * SCALE, -float("inf"))
            if CAUSAL and part == 1:
                score = tl.where(j[None, :] <= i[:, None], score, -float("inf"))
            mx = tl.maximum(mx, tl.max(score, axis=1))
    den = tl.full((BQ,), 0, tl.float32)
    out = tl.full((BQ, 256), 0, tl.float32)
    for part in tl.static_range(2):
        size = L if part == 0 else T
        acc = tl.full((BQ, 256), 0, tl.float32)
        for off in range(0, size, BN):
            j = off + n
            if part == 0:
                slot = tl.load(Tokens + img * L + j, j < L, 0) if PACKED_KV else j
                k = tl.load(kp + slot[:, None] * K2 + d[None, :], (j < L)[:, None], 0)
                v = tl.load(vp + slot[:, None] * V2 + d[None, :], (j < L)[:, None], 0)
                valid = tl.load(Valid + img * MASK_STRIDE + j, j < L, 0)
            else:
                k = tl.load(K + ((start + j) * HK + kh)[:, None] * 256 + d[None, :], (j < length)[:, None], 0)
                v = tl.load(V + (start + j[:, None]) * VS + kh * 256 + d[None, :], (j < length)[:, None], 0)
                valid = j < length
            score = tl.dot(q, tl.trans(k))
            if BANK:
                score = score.to(tl.bfloat16).to(tl.float32)
            weight = tl.where(valid[None, :], tl.exp(score * SCALE - mx[:, None]), 0.0)
            if CAUSAL and part == 1:
                weight = tl.where(j[None, :] <= i[:, None], weight, 0.0)
            den += tl.sum(weight, 1)
            acc = tl.dot(weight.to(tl.bfloat16), v, acc)
        if BANK:
            acc = acc.to(tl.bfloat16).to(tl.float32)
        out += acc
    gate = tl.load(Gate + (start + i[:, None]) * GS + h * GH + d[None, :], (i < length)[:, None], 0).to(tl.float32)
    y = (out / den[:, None]).to(tl.bfloat16).to(tl.float32) * tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    tl.store(O + ((start + i) * HQ + h)[:, None] * 256 + d[None, :], y, (i < length)[:, None])


def branch_attention(q, k, v, gate, kp, vp, valid, scale, blocks, length, causal, offsets=None, images=None, tokens=None):
    n, h, d = q.shape
    out = q.new_zeros((n, h * d))
    _branch_attention[(blocks, h, triton.cdiv(length, 16))](
        q,
        k,
        v,
        gate,
        kp,
        vp,
        valid,
        tokens,
        offsets,
        images,
        out,
        valid.shape[1],
        length,
        h,
        k.shape[1],
        v.stride(0),
        gate.stride(0),
        gate.stride(1),
        *(kp.stride()[:3] if tokens is None else (0, kp.stride(1), kp.stride(0))),
        *(vp.stride()[:3] if tokens is None else (0, vp.stride(1), vp.stride(0))),
        valid.stride(0),
        scale,
        length if offsets is None else 0,
        causal,
        tokens is not None,
        offsets is None,
        16,
        32,
        num_stages=1,
    )
    return out
