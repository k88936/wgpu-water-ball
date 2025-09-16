struct RenderUniforms {
    texel_size: vec2f,
    sphere_size: f32,
    _pad0: f32,
    inv_projection_matrix: mat4x4f,
    projection_matrix: mat4x4f,
    view_matrix: mat4x4f,
    inv_view_matrix: mat4x4f,
}

@group(0) @binding(0) var<uniform> uniforms: RenderUniforms;
@group(0) @binding(1) var sky_sampler: sampler;
@group(0) @binding(2) var sky_cubemap: texture_cube<f32>;

struct VSOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
    // Full-screen triangle
    var p = array<vec2f, 3>(
        vec2f(-1.0, -3.0),
        vec2f(-1.0, 1.0),
        vec2f(3.0, 1.0)
    );
    var out: VSOut;
    out.pos = vec4f(p[vi], 0.0, 1.0);
    out.uv = (p[vi] + 1.0) * 0.5;
    return out;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
    // Reconstruct a view ray direction using inverse projection and inverse view
    let ndc = vec4f(in.uv * 2.0 - 1.0, 1.0, 1.0);
    let view_dir_h = uniforms.inv_projection_matrix * ndc;
    let view_dir = normalize((uniforms.inv_view_matrix * vec4f(view_dir_h.xyz, 0.0)).xyz);
    let color = textureSample(sky_cubemap, sky_sampler, view_dir);
    return vec4f(color.rgb, 1.0);
}
