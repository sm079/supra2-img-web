// Port of torch.randn on the CPU generator (mt19937 + normal_fill), so a seed gives the same
// initial noise as ComfyUI (torch.manual_seed(seed); torch.randn(shape)). The random stream is
// identical; values agree to ~2e-6 (torch's vectorized float log/cos differ in the last ulp).

class MT19937 {
  constructor(seed) {
    this.mt = new Uint32Array(624);
    this.idx = 624;
    this.mt[0] = Number(BigInt.asUintN(32, BigInt(seed)));
    for (let i = 1; i < 624; i++) {
      const p = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      this.mt[i] = (Math.imul(1812433253, p) + i) >>> 0;
    }
  }
  twist() {
    const mt = this.mt;
    for (let i = 0; i < 624; i++) {
      const y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7fffffff);
      let v = mt[(i + 397) % 624] ^ (y >>> 1);
      if (y & 1) v ^= 0x9908b0df;
      mt[i] = v >>> 0;
    }
    this.idx = 0;
  }
  next() {
    if (this.idx >= 624) this.twist();
    let y = this.mt[this.idx++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }
}

export class TorchGenerator {
  constructor(seed) {
    this.mt = new MT19937(seed);
  }
  // at::uniform_real_distribution<float>(0, 1): 24 random bits
  uniform() {
    return Math.fround((this.mt.next() & 0xffffff) / 16777216);
  }
  static fill16(d, o) {
    for (let j = 0; j < 8; j++) {
      const u1 = Math.fround(1 - d[o + j]);
      const u2 = d[o + j + 8];
      const radius = Math.fround(Math.sqrt(Math.fround(-2 * Math.fround(Math.log(u1)))));
      const theta = Math.fround(2.0 * Math.PI * u2);
      d[o + j] = Math.fround(radius * Math.fround(Math.cos(theta)));
      d[o + j + 8] = Math.fround(radius * Math.fround(Math.sin(theta)));
    }
  }
  // torch.randn(n) for float32 with n >= 16 (the vectorized normal_fill path)
  randn(n) {
    const d = new Float32Array(n);
    if (n < 16) throw new Error("randn: only the n >= 16 path is implemented");
    for (let i = 0; i < n; i++) d[i] = this.uniform();
    for (let i = 0; i + 16 <= n; i += 16) TorchGenerator.fill16(d, i);
    if (n % 16 !== 0) {
      const o = n - 16;
      for (let i = 0; i < 16; i++) d[o + i] = this.uniform();
      TorchGenerator.fill16(d, o);
    }
    return d;
  }
}
