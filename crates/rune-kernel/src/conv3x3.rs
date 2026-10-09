pub fn conv3x3_pad1(input: &[f32], weight: &[f32], bias: Option<&[f32]>, out: &mut [f32], c_in: usize, c_out: usize, h: usize, w: usize) {
    for co in 0..c_out {
        let b = if let Some(bb) = bias { bb[co] } else { 0.0 };
        let wbase = co * c_in * 9;
        for oy in 0..h {
            let y0 = oy.wrapping_sub(1);
            for ox in 0..w {
                let x0 = ox.wrapping_sub(1);
                let mut acc = b;
                for ci in 0..c_in {
                    let ibase = ci * h * w;
                    let kbase = wbase + ci * 9;
                    let w0 = weight[kbase];
                    let w1 = weight[kbase + 1];
                    let w2 = weight[kbase + 2];
                    let w3 = weight[kbase + 3];
                    let w4 = weight[kbase + 4];
                    let w5 = weight[kbase + 5];
                    let w6 = weight[kbase + 6];
                    let w7 = weight[kbase + 7];
                    let w8 = weight[kbase + 8];
                    if oy > 0 {
                        let row = ibase + y0 * w;
                        if ox > 0 {
                            acc += input[row + x0] * w0;
                        }
                        acc += input[row + ox] * w1;
                        if ox + 1 < w {
                            acc += input[row + ox + 1] * w2;
                        }
                    }
                    {
                        let row = ibase + oy * w;
                        if ox > 0 {
                            acc += input[row + x0] * w3;
                        }
                        acc += input[row + ox] * w4;
                        if ox + 1 < w {
                            acc += input[row + ox + 1] * w5;
                        }
                    }
                    if oy + 1 < h {
                        let row = ibase + (oy + 1) * w;
                        if ox > 0 {
                            acc += input[row + x0] * w6;
                        }
                        acc += input[row + ox] * w7;
                        if ox + 1 < w {
                            acc += input[row + ox + 1] * w8;
                        }
                    }
                }
                out[co * h * w + oy * w + ox] = acc;
            }
        }
    }
}

pub fn conv3x3_pad1_relu(input: &[f32], weight: &[f32], bias: Option<&[f32]>, out: &mut [f32], c_in: usize, c_out: usize, h: usize, w: usize) {
    for co in 0..c_out {
        let b = if let Some(bb) = bias { bb[co] } else { 0.0 };
        let wbase = co * c_in * 9;
        for oy in 0..h {
            let y0 = oy.wrapping_sub(1);
            for ox in 0..w {
                let x0 = ox.wrapping_sub(1);
                let mut acc = b;
                for ci in 0..c_in {
                    let ibase = ci * h * w;
                    let kbase = wbase + ci * 9;
                    let w0 = weight[kbase];
                    let w1 = weight[kbase + 1];
                    let w2 = weight[kbase + 2];
                    let w3 = weight[kbase + 3];
                    let w4 = weight[kbase + 4];
                    let w5 = weight[kbase + 5];
                    let w6 = weight[kbase + 6];
                    let w7 = weight[kbase + 7];
                    let w8 = weight[kbase + 8];
                    if oy > 0 {
                        let row = ibase + y0 * w;
                        if ox > 0 {
                            acc += input[row + x0] * w0;
                        }
                        acc += input[row + ox] * w1;
                        if ox + 1 < w {
                            acc += input[row + ox + 1] * w2;
                        }
                    }
                    {
                        let row = ibase + oy * w;
                        if ox > 0 {
                            acc += input[row + x0] * w3;
                        }
                        acc += input[row + ox] * w4;
                        if ox + 1 < w {
                            acc += input[row + ox + 1] * w5;
                        }
                    }
                    if oy + 1 < h {
                        let row = ibase + (oy + 1) * w;
                        if ox > 0 {
                            acc += input[row + x0] * w6;
                        }
                        acc += input[row + ox] * w7;
                        if ox + 1 < w {
                            acc += input[row + ox + 1] * w8;
                        }
                    }
                }
                out[co * h * w + oy * w + ox] = if acc < 0.0 { 0.0 } else { acc };
            }
        }
    }
}
