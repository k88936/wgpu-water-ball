use std::borrow::Cow;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

// === Uniforms and data structs (CPU-side) ===

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct RenderUniforms {
    texel_size: [f32; 2],
    sphere_size: f32,
    _pad0: f32, // padding to 16-byte alignment
    inv_projection_matrix: [[f32; 4]; 4],
    projection_matrix: [[f32; 4]; 4],
    view_matrix: [[f32; 4]; 4],
    inv_view_matrix: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MouseInfoUniform {
    screen_size: [f32; 2],
    mouse_coord: [f32; 2],
    mouse_vel: [f32; 2],
    mouse_radius: f32,
    _pad: f32, // pad to 32 bytes total
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Vec3F32([f32; 3], f32); // helper for 16-byte stride packing if needed

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ParticleCPU {
    // Matches WGSL Particle: position: vec3f, v: vec3f, C: mat3x3f (padded to 80 bytes total)
    position: [f32; 3],
    _pad0: f32,
    v: [f32; 3],
    _pad1: f32,
    // 3x3 matrix laid as 3 vec4 (with last component padding) for 16-byte stride alignment
    c0: [f32; 4],
    c1: [f32; 4],
    c2: [f32; 4],
}

impl ParticleCPU {
    fn new(pos: [f32; 3], vel: [f32; 3]) -> Self {
        Self {
            position: pos,
            _pad0: 0.0,
            v: vel,
            _pad1: 0.0,
            c0: [0.0, 0.0, 0.0, 0.0],
            c1: [0.0, 0.0, 0.0, 0.0],
            c2: [0.0, 0.0, 0.0, 0.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct PosVelCPU {
    position: [f32; 3],
    _pad0: f32,
    v: [f32; 3],
    density: f32,
}

// === WGSL sources (linked from WaterBall) ===

const CLEAR_GRID_WGSL: &str = include_str!("../WaterBall/mls-mpm/clearGrid.wgsl");
const SPAWN_WGSL: &str = include_str!("../WaterBall/mls-mpm/spawnParticles.wgsl");
const P2G1_WGSL_RAW: &str = include_str!("../WaterBall/mls-mpm/p2g_1.wgsl");
const P2G2_WGSL_RAW: &str = include_str!("../WaterBall/mls-mpm/p2g_2.wgsl");
const UPDATE_GRID_WGSL_RAW: &str = include_str!("../WaterBall/mls-mpm/updateGrid.wgsl");
const G2P_WGSL_RAW: &str = include_str!("../WaterBall/mls-mpm/g2p.wgsl");
const COPY_POS_WGSL: &str = include_str!("../WaterBall/mls-mpm/copyPosition.wgsl");

const SPHERE_WGSL_RAW: &str = include_str!("../WaterBall/render/sphere.wgsl");

// === App State ===

struct Pipelines {
    clear_grid: wgpu::ComputePipeline,
    spawn: wgpu::ComputePipeline,
    p2g1: wgpu::ComputePipeline,
    p2g2: wgpu::ComputePipeline,
    update_grid: wgpu::ComputePipeline,
    g2p: wgpu::ComputePipeline,
    copy_pos: wgpu::ComputePipeline,
    sphere_pipeline: wgpu::RenderPipeline,
}

struct GpuBuffers {
    particle_buffer: wgpu::Buffer,     // Particle (storage)
    cell_buffer: wgpu::Buffer,         // Grid cells (storage)
    density_buffer: wgpu::Buffer,      // Density per particle (storage)
    posvel_buffer: wgpu::Buffer,       // Rendered positions/velocities/density (storage)
    render_uniform: wgpu::Buffer,      // Render uniforms
    init_box_size: wgpu::Buffer,       // vec3f uniform
    real_box_size: wgpu::Buffer,       // vec3f uniform
    num_particles: wgpu::Buffer,       // u32 uniform
    mouse_uniform: wgpu::Buffer,       // MouseInfo
    sphere_radius: wgpu::Buffer,       // f32 uniform
}

struct Textures {
    depth_test_view: wgpu::TextureView, // depth32float
    depth_map_view: wgpu::TextureView,  // r32float second color att
}

struct BindGroups {
    clear_grid: wgpu::BindGroup,
    spawn: wgpu::BindGroup,
    p2g1: wgpu::BindGroup,
    p2g2: wgpu::BindGroup,
    update_grid: wgpu::BindGroup,
    g2p: wgpu::BindGroup,
    copy_pos: wgpu::BindGroup,
    sphere: wgpu::BindGroup,
}

struct SimConfig {
    max_x: u32,
    max_y: u32,
    max_z: u32,
    num_particles: u32,
    render_diameter: f32,
    dt: f32,
    fixed_point_multiplier: f32,
    stiffness: f32,
    rest_density: f32,
    dynamic_viscosity: f32,
    sphere_radius: f32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            max_x: 80,
            max_y: 80,
            max_z: 80,
            num_particles: 30_000, // start modestly; can scale up
            render_diameter: 1.4, // matches TS 2 * radius where radius=0.7 in NDC units for billboard size
            dt: 0.20,
            fixed_point_multiplier: 1e7,
            stiffness: 3.0,
            rest_density: 4.0,
            dynamic_viscosity: 0.1,
            sphere_radius: 20.0, // world units approx
        }
    }
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_format: wgpu::TextureFormat,
    size: winit::dpi::PhysicalSize<u32>,

    pipelines: Option<Pipelines>,
    buffers: Option<GpuBuffers>,
    binds: Option<BindGroups>,
    textures: Option<Textures>,

    sim_cfg: SimConfig,
    grid_count: u32,
    init_box_size_cpu: [f32; 3],
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).expect("create surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("request_device");

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let mut this = Self {
            window,
            surface,
            device,
            queue,
            surface_format,
            size,
            pipelines: None,
            buffers: None,
            binds: None,
            textures: None,
            sim_cfg: SimConfig::default(),
            grid_count: 0,
            init_box_size_cpu: [60.0, 60.0, 60.0],
        };

        this.configure_surface();
        this.init_resources().await;

        this
    }

    fn configure_surface(&self) {
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: self.surface_format,
            view_formats: vec![self.surface_format.add_srgb_suffix()],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: self.size.width,
            height: self.size.height,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::AutoVsync,
        };
        self.surface.configure(&self.device, &config);
    }

    fn make_shader_module(&self, code: &str) -> wgpu::ShaderModule {
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(code)) })
    }

    fn inject_overrides(code: &str, overrides: &[(&str, f32)]) -> String {
        // Replace lines like: override name: f32; -> override name: f32 = value;
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

    async fn init_resources(&mut self) {
        let SimConfig {
            max_x,
            max_y,
            max_z,
            num_particles,
            render_diameter,
            dt,
            fixed_point_multiplier,
            stiffness,
            rest_density,
            dynamic_viscosity,
            sphere_radius,
        } = self.sim_cfg;

        // Grid and counts
        let init_box_size = [60.0_f32, 60.0, 60.0];
        let real_box_size = init_box_size;
        let grid_count = (init_box_size[0].ceil() as u32)
            * (init_box_size[1].ceil() as u32)
            * (init_box_size[2].ceil() as u32);
    self.grid_count = grid_count;
    self.init_box_size_cpu = init_box_size;

        // Buffers
        let max_grid_count = (max_x * max_y * max_z) as usize;
        let cell_struct_size = 16u64; // 4*i32
        let cell_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cells buffer"),
            size: cell_struct_size * max_grid_count as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particle_stride = 80u64; // mlsmpmParticleStructSize
        let particle_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles buffer"),
            size: particle_stride * (num_particles as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let posvel_stride = 32u64; // vec3 + pad + vec3 + f32
        let posvel_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("posvel buffer"),
            size: posvel_stride * (num_particles as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let density_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("density buffer"),
            size: 4 * (num_particles as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let init_box_size_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("init_box_size"),
            contents: bytemuck::bytes_of(&init_box_size),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let real_box_size_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("real_box_size"),
            contents: bytemuck::bytes_of(&real_box_size),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let num_particles_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("num_particles"),
            contents: bytemuck::bytes_of(&(num_particles as u32)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let mouse_uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mouse_uniform"),
            size: std::mem::size_of::<MouseInfoUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sphere_radius_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sphere_radius"),
            contents: bytemuck::bytes_of(&sphere_radius),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Render uniforms
        let aspect = self.size.width.max(1) as f32 / self.size.height.max(1) as f32;
        let fov = 45.0_f32.to_radians();
        let proj = Mat4::perspective_rh(fov, aspect, 0.1, 300.0);
        let view = Mat4::look_at_rh(
            Vec3::new(init_box_size[0] * 0.5, init_box_size[1] * 0.5, init_box_size[2] * 0.5 + 70.0),
            Vec3::new(init_box_size[0] * 0.5, init_box_size[1] * 0.5, init_box_size[2] * 0.5),
            Vec3::Y,
        );
        let uniforms = RenderUniforms {
            texel_size: [1.0 / self.size.width.max(1) as f32, 1.0 / self.size.height.max(1) as f32],
            sphere_size: render_diameter,
            _pad0: 0.0,
            inv_projection_matrix: proj.inverse().to_cols_array_2d(),
            projection_matrix: proj.to_cols_array_2d(),
            view_matrix: view.to_cols_array_2d(),
            inv_view_matrix: view.inverse().to_cols_array_2d(),
        };
        let render_uniform = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render_uniform"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Initialize particles on CPU (dam-break like)
        let mut particles: Vec<ParticleCPU> = Vec::with_capacity(num_particles as usize);
        let spacing = 0.55_f32;
        'outer: for j in (3..((init_box_size[1] * 0.8) as i32)).map(|v| v as f32).step_by(1) {
            let mut y = j;
            while y < init_box_size[1] - 4.0 {
                let mut i = 3.0_f32;
                while i < init_box_size[0] - 4.0 {
                    let mut k = 3.0_f32;
                    while k < init_box_size[2] * 0.5 {
                        if particles.len() >= num_particles as usize {
                            break 'outer;
                        }
                        let jitter = 2.0 * 0.0_f32; // deterministic first; can add rand later
                        particles.push(ParticleCPU::new(
                            [i + jitter, y + jitter, k + jitter],
                            [0.0, 0.0, 0.0],
                        ));
                        k += spacing;
                    }
                    i += spacing;
                }
                y += spacing;
            }
        }
        if !particles.is_empty() {
            self.queue.write_buffer(&particle_buffer, 0, bytemuck::cast_slice(&particles));
        }

        // Textures (depth-test, depth-map second color attachment)
        let depth_test_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth32float"),
            size: wgpu::Extent3d { width: self.size.width.max(1), height: self.size.height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_test_view = depth_test_tex.create_view(&Default::default());

        let depth_map_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depthmap_r32float"),
            size: wgpu::Extent3d { width: self.size.width.max(1), height: self.size.height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_map_view = depth_map_tex.create_view(&Default::default());

        // Shader modules
        let clear_grid_mod = self.make_shader_module(CLEAR_GRID_WGSL);
        let spawn_mod = self.make_shader_module(SPAWN_WGSL);
        let p2g1_src = Self::inject_overrides(P2G1_WGSL_RAW, &[("fixed_point_multiplier", fixed_point_multiplier)]);
        let p2g2_src = Self::inject_overrides(
            P2G2_WGSL_RAW,
            &[
                ("fixed_point_multiplier", fixed_point_multiplier),
                ("stiffness", stiffness),
                ("rest_density", rest_density),
                ("dynamic_viscosity", dynamic_viscosity),
                ("dt", dt),
            ],
        );
        let update_grid_src = Self::inject_overrides(
            UPDATE_GRID_WGSL_RAW,
            &[("fixed_point_multiplier", fixed_point_multiplier), ("dt", dt)],
        );
        let g2p_src = Self::inject_overrides(G2P_WGSL_RAW, &[("fixed_point_multiplier", fixed_point_multiplier), ("dt", dt)]);
        let mut sphere_src = Self::inject_overrides(SPHERE_WGSL_RAW, &[("restDensity", rest_density), ("densitySizeScale", 4.0)]);
        sphere_src = sphere_src.replace(
            "let size = uniforms.sphere_size * clamp(particles[instance_index].density / restDensity * densitySizeScale, 0.0, 1.0);",
            "let size = uniforms.sphere_size;",
        );

        let p2g1_mod = self.make_shader_module(&p2g1_src);
        let p2g2_mod = self.make_shader_module(&p2g2_src);
        let update_grid_mod = self.make_shader_module(&update_grid_src);
        let g2p_mod = self.make_shader_module(&g2p_src);
        let copy_pos_mod = self.make_shader_module(COPY_POS_WGSL);
    let sphere_mod = self.make_shader_module(&sphere_src);

        // Compute pipelines with constant overrides
        let make_cp = |label: &str, module: &wgpu::ShaderModule, entry: &str| {
            self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        let clear_grid = make_cp("clearGrid", &clear_grid_mod, "clearGrid");
        let spawn = make_cp("spawn", &spawn_mod, "spawn");
        let p2g1 = make_cp("p2g_1", &p2g1_mod, "p2g_1");
        let p2g2 = make_cp("p2g_2", &p2g2_mod, "p2g_2");
        let update_grid = make_cp("updateGrid", &update_grid_mod, "updateGrid");
        let g2p = make_cp("g2p", &g2p_mod, "g2p");
        let copy_pos = make_cp("copyPosition", &copy_pos_mod, "copyPosition");

        // Sphere render pipeline (2 color attachments: swapchain, r32float) + depth
        let sphere_pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sphere_pipeline"),
            layout: None,
            vertex: wgpu::VertexState { module: &sphere_mod, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState {
                module: &sphere_mod,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState { format: self.surface_format, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::R32Float, blend: None, write_mask: wgpu::ColorWrites::RED }),
                ],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        // Bind groups
        let clear_grid_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clearGridBG"),
            layout: &clear_grid.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: cell_buffer.as_entire_binding() }],
        });
        let spawn_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spawnBG"),
            layout: &spawn.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: init_box_size_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: num_particles_buf.as_entire_binding() },
            ],
        });
        let p2g1_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("p2g1BG"),
            layout: &p2g1.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cell_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: init_box_size_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: num_particles_buf.as_entire_binding() },
            ],
        });
        let p2g2_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("p2g2BG"),
            layout: &p2g2.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cell_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: init_box_size_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: num_particles_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: density_buffer.as_entire_binding() },
            ],
        });
        let update_grid_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("updateGridBG"),
            layout: &update_grid.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: cell_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: real_box_size_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: init_box_size_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: render_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&depth_map_view) },
                wgpu::BindGroupEntry { binding: 5, resource: mouse_uniform.as_entire_binding() },
            ],
        });
        let g2p_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("g2pBG"),
            layout: &g2p.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cell_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: real_box_size_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: init_box_size_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: num_particles_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: sphere_radius_buf.as_entire_binding() },
            ],
        });
        let copy_pos_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("copyPosBG"),
            layout: &copy_pos.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: posvel_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: num_particles_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: density_buffer.as_entire_binding() },
            ],
        });
        let sphere_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sphereBG"),
            layout: &sphere_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: posvel_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: render_uniform.as_entire_binding() },
                // binding(2) stretchStrength: f32 -> supply via a tiny 4B uniform buffer set to 2.0
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("stretchStrength"),
                            contents: bytemuck::bytes_of(&2.0_f32),
                            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        })
                        .as_entire_binding(),
                },
            ],
        });

        // Store resources
        self.pipelines = Some(Pipelines { clear_grid, spawn, p2g1, p2g2, update_grid, g2p, copy_pos, sphere_pipeline });
        self.buffers = Some(GpuBuffers {
            particle_buffer,
            cell_buffer,
            density_buffer,
            posvel_buffer,
            render_uniform,
            init_box_size: init_box_size_buf,
            real_box_size: real_box_size_buf,
            num_particles: num_particles_buf,
            mouse_uniform,
            sphere_radius: sphere_radius_buf,
        });
        self.textures = Some(Textures { depth_test_view, depth_map_view });
        self.binds = Some(BindGroups { clear_grid: clear_grid_bg, spawn: spawn_bg, p2g1: p2g1_bg, p2g2: p2g2_bg, update_grid: update_grid_bg, g2p: g2p_bg, copy_pos: copy_pos_bg, sphere: sphere_bg });

        ()
    }

    fn update_mouse_uniform(&self, mouse: MouseInfoUniform) {
        if let Some(b) = &self.buffers {
            self.queue.write_buffer(&b.mouse_uniform, 0, bytemuck::bytes_of(&mouse));
        }
    }

    fn step_simulation(&mut self) {
        // Build command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sim encoder") });

        // Compute pass sequence
        if let (Some(pipes), Some(buffs), Some(binds)) = (&self.pipelines, &self.buffers, &self.binds) {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("mpm"), timestamp_writes: None });

            let particles = self.sim_cfg.num_particles;
            let grids = self.grid_count;

            // Clear grid
            cpass.set_pipeline(&pipes.clear_grid);
            cpass.set_bind_group(0, &binds.clear_grid, &[]);
            cpass.dispatch_workgroups((grids + 63) / 64, 1, 1);

            // P2G
            cpass.set_pipeline(&pipes.p2g1);
            cpass.set_bind_group(0, &binds.p2g1, &[]);
            cpass.dispatch_workgroups((particles + 63) / 64, 1, 1);

            cpass.set_pipeline(&pipes.p2g2);
            cpass.set_bind_group(0, &binds.p2g2, &[]);
            cpass.dispatch_workgroups((particles + 63) / 64, 1, 1);

            // Grid update
            cpass.set_pipeline(&pipes.update_grid);
            cpass.set_bind_group(0, &binds.update_grid, &[]);
            cpass.dispatch_workgroups((grids + 63) / 64, 1, 1);

            // G2P
            cpass.set_pipeline(&pipes.g2p);
            cpass.set_bind_group(0, &binds.g2p, &[]);
            cpass.dispatch_workgroups((particles + 63) / 64, 1, 1);

            // Copy to render buffer
            cpass.set_pipeline(&pipes.copy_pos);
            cpass.set_bind_group(0, &binds.copy_pos, &[]);
            cpass.dispatch_workgroups((particles + 63) / 64, 1, 1);
        }

        // Render pass: spheres to swapchain + r32float depth map
        let surface_tex = match self.surface.get_current_texture() {
            Ok(st) => st,
            Err(e) => {
                eprintln!("Surface error: {e:?}, reconfiguring");
                self.configure_surface();
                self.surface.get_current_texture().expect("failed to acquire surface texture")
            }
        };
        let surface_view = surface_tex.texture.create_view(&Default::default());

        if let (Some(pipes), Some(textures)) = (&self.pipelines, &self.textures) {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sphere pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment { view: &surface_view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.03, a: 1.0 }), store: wgpu::StoreOp::Store } }),
                    Some(wgpu::RenderPassColorAttachment { view: &textures.depth_map_view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 1e6, g: 0.0, b: 0.0, a: 1.0 }), store: wgpu::StoreOp::Store } }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &textures.depth_test_view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&pipes.sphere_pipeline);
            if let Some(binds) = &self.binds { rpass.set_bind_group(0, &binds.sphere, &[]); }
            // 6 vertices per quad, instancing used per particle
            rpass.draw(0..6, 0..self.sim_cfg.num_particles);
        }

        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        surface_tex.present();
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.configure_surface();
        // Recreate size-dependent textures
        let depth_test_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth32float"),
            size: wgpu::Extent3d { width: self.size.width.max(1), height: self.size.height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_map_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depthmap_r32float"),
            size: wgpu::Extent3d { width: self.size.width.max(1), height: self.size.height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        if let Some(textures) = &mut self.textures {
            textures.depth_test_view = depth_test_tex.create_view(&Default::default());
            textures.depth_map_view = depth_map_tex.create_view(&Default::default());
        }

        // Update texel size and camera in uniforms
        let aspect = self.size.width as f32 / self.size.height as f32;
        let fov = 45.0_f32.to_radians();
        let proj = Mat4::perspective_rh(fov, aspect, 0.1, 300.0);
        let center = Vec3::new(
            self.init_box_size_cpu[0] * 0.5,
            self.init_box_size_cpu[1] * 0.5,
            self.init_box_size_cpu[2] * 0.5,
        );
        let cam_pos = center + Vec3::new(0.0, 0.0, 70.0);
        let view = Mat4::look_at_rh(cam_pos, center, Vec3::Y);
        let uniforms = RenderUniforms {
            texel_size: [1.0 / self.size.width.max(1) as f32, 1.0 / self.size.height.max(1) as f32],
            sphere_size: self.sim_cfg.render_diameter,
            _pad0: 0.0,
            inv_projection_matrix: proj.inverse().to_cols_array_2d(),
            projection_matrix: proj.to_cols_array_2d(),
            view_matrix: view.to_cols_array_2d(),
            inv_view_matrix: view.inverse().to_cols_array_2d(),
        };
    if let Some(b) = &self.buffers { self.queue.write_buffer(&b.render_uniform, 0, bytemuck::bytes_of(&uniforms)); }
    }

    fn get_window(&self) -> &Window { &self.window }
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(event_loop.create_window(Window::default_attributes()).unwrap());
    let state = pollster::block_on(State::new(window.clone()));
        self.state = Some(state);
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let state = match self.state.as_mut() { Some(s) => s, None => return };
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                state.resize(size);
            }
            WindowEvent::RedrawRequested => {
                // Update mouse uniform with no input for now
                let mouse = MouseInfoUniform { screen_size: [state.size.width as f32, state.size.height as f32], mouse_coord: [0.5, 0.5], mouse_vel: [0.0, 0.0], mouse_radius: 6.0, _pad: 0.0 };
                state.update_mouse_uniform(mouse);
                state.step_simulation();
                state.get_window().request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
