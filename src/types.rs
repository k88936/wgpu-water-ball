//! Shared data structures, uniforms, and configuration types for the simulation and renderer.
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use winit::dpi::PhysicalSize;

/// Per-frame render uniforms consumed by both compute (for some transforms) and render pipelines.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RenderUniforms {
    pub texel_size: [f32; 2],
    pub sphere_size: f32,
    _pad0: f32, // padding to 16-byte alignment
    pub inv_projection_matrix: [[f32; 4]; 4],
    pub projection_matrix: [[f32; 4]; 4],
    pub view_matrix: [[f32; 4]; 4],
    pub inv_view_matrix: [[f32; 4]; 4],
}

/// Mouse interaction uniform (currently mostly placeholder, but structured for future input).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MouseInfoUniform {
    pub screen_size: [f32; 2],
    pub mouse_coord: [f32; 2],
    pub mouse_vel: [f32; 2],
    pub mouse_radius: f32,
    pub _pad: f32, // pad to 32 bytes total
}

/// Helper wrapper enabling 16-byte aligned vec3 storage (currently unused but retained for parity).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vec3F32(pub [f32; 3], pub f32);

/// CPU-side representation of a particle matching the WGSL layout (80 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ParticleCPU {
    pub position: [f32; 3],
    _pad0: f32,
    pub v: [f32; 3],
    _pad1: f32,
    pub c0: [f32; 4],
    pub c1: [f32; 4],
    pub c2: [f32; 4],
}

impl ParticleCPU {
    pub fn new(pos: [f32; 3], vel: [f32; 3]) -> Self {
        Self {
            position: pos,
            _pad0: 0.0,
            v: vel,
            _pad1: 0.0,
            c0: [0.0; 4],
            c1: [0.0; 4],
            c2: [0.0; 4],
        }
    }
}

/// Compact position/velocity/density struct used for rendering billboards.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PosVelCPU {
    pub position: [f32; 3],
    _pad0: f32,
    pub v: [f32; 3],
    pub density: f32,
}

/// Simulation configuration parameters.
#[derive(Clone, Copy, Debug)]
pub struct SimConfig {
    pub max_x: u32,
    pub max_y: u32,
    pub max_z: u32,
    pub num_particles: u32,
    pub render_diameter: f32,
    pub dt: f32,
    pub fixed_point_multiplier: f32,
    pub stiffness: f32,
    pub rest_density: f32,
    pub dynamic_viscosity: f32,
    pub sphere_radius: f32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            max_x: 80,
            max_y: 80,
            max_z: 80,
            num_particles: 30_000,
            render_diameter: 1.4,
            dt: 0.20,
            fixed_point_multiplier: 1e7,
            stiffness: 3.0,
            rest_density: 4.0,
            dynamic_viscosity: 0.1,
            sphere_radius: 20.0,
        }
    }
}

/// Utility: inject `override name: f32 = value;` into WGSL source for provided overrides.
pub fn inject_overrides(code: &str, overrides: &[(&str, f32)]) -> String {
    let mut s = code.to_string();
    for (name, val) in overrides.iter() {
        let needle = format!("override {name}: f32;");
        if s.contains(&needle) {
            let repl = format!("override {name}: f32 = {val};");
            s = s.replace(&needle, &repl);
        }
    }
    s
}

/// Build the initial (or resized) camera + projection uniforms.
pub fn build_render_uniforms(size: PhysicalSize<u32>, init_box_size: [f32; 3], sphere_size: f32) -> RenderUniforms {
    let aspect = size.width.max(1) as f32 / size.height.max(1) as f32;
    let fov = 45.0_f32.to_radians();
    let proj = Mat4::perspective_rh(fov, aspect, 0.1, 300.0);
    let center = Vec3::new(
        init_box_size[0] * 0.5,
        init_box_size[1] * 0.5,
        init_box_size[2] * 0.5,
    );
    let cam_pos = center + Vec3::new(0.0, 0.0, 70.0);
    let view = Mat4::look_at_rh(cam_pos, center, Vec3::Y);
    RenderUniforms {
        texel_size: [
            1.0 / size.width.max(1) as f32,
            1.0 / size.height.max(1) as f32,
        ],
        sphere_size,
        _pad0: 0.0,
        inv_projection_matrix: proj.inverse().to_cols_array_2d(),
        projection_matrix: proj.to_cols_array_2d(),
        view_matrix: view.to_cols_array_2d(),
        inv_view_matrix: view.inverse().to_cols_array_2d(),
    }
}
