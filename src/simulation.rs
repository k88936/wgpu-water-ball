//! Simulation module: owns compute pipelines, buffers, bind groups, and stepping logic.
use std::borrow::Cow;
use wgpu::include_wgsl;
use wgpu::util::DeviceExt;

use crate::types::{
    build_render_uniforms, inject_overrides, MouseInfoUniform, ParticleCPU, RenderUniforms, SimConfig,
};

// WGSL includes (shared with original project layout)
const CLEAR_GRID_WGSL: &str = include_str!("../WaterBall/mls-mpm/clearGrid.wgsl");
const SPAWN_WGSL: &str = include_str!("../WaterBall/mls-mpm/spawnParticles.wgsl");
const P2G1_WGSL_RAW: &str = include_str!("../WaterBall/mls-mpm/p2g_1.wgsl");
const P2G2_WGSL_RAW: &str = include_str!("../WaterBall/mls-mpm/p2g_2.wgsl");
const UPDATE_GRID_WGSL_RAW: &str = include_str!("../WaterBall/mls-mpm/updateGrid.wgsl");
const G2P_WGSL_RAW: &str = include_str!("../WaterBall/mls-mpm/g2p.wgsl");
const COPY_POS_WGSL: &str = include_str!("../WaterBall/mls-mpm/copyPosition.wgsl");

/// Buffers needed across compute stages and for rendering.
#[allow(dead_code)]
pub struct GpuBuffers {
    pub particle_buffer: wgpu::Buffer,
    pub cell_buffer: wgpu::Buffer,
    pub density_buffer: wgpu::Buffer,
    pub posvel_buffer: wgpu::Buffer,
    pub render_uniform: wgpu::Buffer,
    pub init_box_size: wgpu::Buffer,
    pub real_box_size: wgpu::Buffer,
    pub num_particles: wgpu::Buffer,
    pub mouse_uniform: wgpu::Buffer,
    pub sphere_radius: wgpu::Buffer,
}

#[allow(dead_code)]
struct Pipelines {
    clear_grid: wgpu::ComputePipeline,
    spawn: wgpu::ComputePipeline,
    p2g1: wgpu::ComputePipeline,
    p2g2: wgpu::ComputePipeline,
    update_grid: wgpu::ComputePipeline,
    g2p: wgpu::ComputePipeline,
    copy_pos: wgpu::ComputePipeline,
}

#[allow(dead_code)]
struct BindGroups {
    clear_grid: wgpu::BindGroup,
    spawn: wgpu::BindGroup,
    p2g1: wgpu::BindGroup,
    p2g2: wgpu::BindGroup,
    g2p: wgpu::BindGroup,
    copy_pos: wgpu::BindGroup,
}

/// Simulation owns all compute resources and can provide handles required by the renderer.
pub struct Simulation {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub config: SimConfig,
    pub init_box_size_cpu: [f32; 3],
    grid_count: u32,
    pipelines: Pipelines,
    bind_groups: BindGroups,
    pub buffers: GpuBuffers,
}

impl Simulation {
    pub async fn new(adapter: &wgpu::Adapter, config: SimConfig, size: winit::dpi::PhysicalSize<u32>) -> Result<Self, wgpu::RequestDeviceError> {
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("sim-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                trace: wgpu::Trace::Off,
            })
            .await?;

        // Box + grid sizing
        let init_box_size = [60.0_f32, 60.0, 60.0];
        let real_box_size = init_box_size;
        let grid_count = (init_box_size[0].ceil() as u32)
            * (init_box_size[1].ceil() as u32)
            * (init_box_size[2].ceil() as u32);

        /*
        The buffers work together in this sequence:
        1. **Particle Buffer** → P2G shaders → **Cell Buffer** (mass/velocity accumulation)
        2. **Cell Buffer** → P2G2 shader → **Density Buffer** (density calculation)
        3. **Cell Buffer** → UpdateGrid shader (force application)
        4. **Cell Buffer** + **Particle Buffer** → G2P shader → Updated **Particle Buffer**
        5. **Particle Buffer** → CopyPosition shader → **PosVel Buffer** (for rendering)

        This architecture keeps all simulation data on the GPU for maximum performance while maintaining the data dependencies required by the MPM algorithm.

         */
        // Buffers
        let max_grid_count = (config.max_x * config.max_y * config.max_z) as usize;
        let cell_struct_size = 16u64; // 4*i32
        let cell_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cells buffer"),
            size: cell_struct_size * max_grid_count as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particle_stride = 80u64;
        let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles buffer"),
            size: particle_stride * (config.num_particles as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let posvel_stride = 32u64;
        let posvel_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("posvel buffer"),
            size: posvel_stride * (config.num_particles as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let density_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("density buffer"),
            size: 4 * (config.num_particles as u64),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let init_box_size_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("init_box_size"),
            contents: bytemuck::bytes_of(&init_box_size),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let real_box_size_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("real_box_size"),
            contents: bytemuck::bytes_of(&real_box_size),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let num_particles_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("num_particles"),
            contents: bytemuck::bytes_of(&(config.num_particles as u32)),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let mouse_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mouse_uniform"),
            size: std::mem::size_of::<MouseInfoUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sphere_radius_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sphere_radius"),
            contents: bytemuck::bytes_of(&config.sphere_radius),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Render uniforms buffer
        let uniforms: RenderUniforms = build_render_uniforms(size, init_box_size, config.render_diameter);
        let render_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("render_uniform"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Initialize CPU particles
        let mut particles: Vec<ParticleCPU> = Vec::with_capacity(config.num_particles as usize);
        let spacing = 0.55_f32;
        'outer: for j in (3..((init_box_size[1] * 0.8) as i32)).map(|v| v as f32).step_by(1) {
            let mut y = j;
            while y < init_box_size[1] - 4.0 {
                let mut i = 3.0_f32;
                while i < init_box_size[0] - 4.0 {
                    let mut k = 3.0_f32;
                    while k < init_box_size[2] * 0.5 {
                        if particles.len() >= config.num_particles as usize { break 'outer; }
                        particles.push(ParticleCPU::new([i, y, k], [0.0, 0.0, 0.0]));
                        k += spacing;
                    }
                    i += spacing;
                }
                y += spacing;
            }
        }
        if !particles.is_empty() {
            queue.write_buffer(&particle_buffer, 0, bytemuck::cast_slice(&particles));
        }

        // Shader modules
        let make_shader = |code: &str| device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(code)),
        });

        let clear_grid_mod = make_shader(CLEAR_GRID_WGSL);
        let spawn_mod = make_shader(SPAWN_WGSL);
        let p2g1_mod = make_shader(&inject_overrides(P2G1_WGSL_RAW, &[("fixed_point_multiplier", config.fixed_point_multiplier)]));
        let p2g2_mod = make_shader(&inject_overrides(P2G2_WGSL_RAW, &[("fixed_point_multiplier", config.fixed_point_multiplier), ("stiffness", config.stiffness), ("rest_density", config.rest_density), ("dynamic_viscosity", config.dynamic_viscosity), ("dt", config.dt)]));
        let update_grid_mod = make_shader(&inject_overrides(UPDATE_GRID_WGSL_RAW, &[("fixed_point_multiplier", config.fixed_point_multiplier), ("dt", config.dt)]));
        let g2p_mod = make_shader(&inject_overrides(G2P_WGSL_RAW, &[("fixed_point_multiplier", config.fixed_point_multiplier), ("dt", config.dt)]));
        let copy_pos_mod = make_shader(COPY_POS_WGSL);

        let make_cp = |label: &str, module: &wgpu::ShaderModule, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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

        // Bind groups
        let clear_grid_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clearGridBG"),
            layout: &clear_grid.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: cell_buffer.as_entire_binding() }],
        });
        let spawn_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spawnBG"),
            layout: &spawn.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: init_box_size_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: num_particles_buf.as_entire_binding() },
            ],
        });
        let p2g1_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("p2g1BG"),
            layout: &p2g1.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: cell_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: init_box_size_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: num_particles_buf.as_entire_binding() },
            ],
        });
        let p2g2_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        let g2p_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        let copy_pos_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("copyPosBG"),
            layout: &copy_pos.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: posvel_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: num_particles_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: density_buffer.as_entire_binding() },
            ],
        });

        Ok(Self {
            device,
            queue,
            config,
            init_box_size_cpu: init_box_size,
            grid_count,
            pipelines: Pipelines { clear_grid, spawn, p2g1, p2g2, update_grid, g2p, copy_pos },
            bind_groups: BindGroups { clear_grid: clear_grid_bg, spawn: spawn_bg, p2g1: p2g1_bg, p2g2: p2g2_bg, g2p: g2p_bg, copy_pos: copy_pos_bg },
            buffers: GpuBuffers { particle_buffer, cell_buffer, density_buffer, posvel_buffer, render_uniform, init_box_size: init_box_size_buf, real_box_size: real_box_size_buf, num_particles: num_particles_buf, mouse_uniform, sphere_radius: sphere_radius_buf },
        })
    }

    pub fn device(&self) -> &wgpu::Device { &self.device }
    pub fn queue(&self) -> &wgpu::Queue { &self.queue }

    /// Update mouse uniform data.
    pub fn update_mouse(&self, mouse: MouseInfoUniform) {
        self.queue.write_buffer(&self.buffers.mouse_uniform, 0, bytemuck::bytes_of(&mouse));
    }

    /// Rebuild render uniforms (e.g., after resize) and upload.
    pub fn update_render_uniforms(&self, size: winit::dpi::PhysicalSize<u32>) {
        let uniforms = build_render_uniforms(size, self.init_box_size_cpu, self.config.render_diameter);
        self.queue.write_buffer(&self.buffers.render_uniform, 0, bytemuck::bytes_of(&uniforms));
    }

    /// Execute compute pipeline sequence producing updated particle buffers.
    pub fn step(&self, depth_map_view: &wgpu::TextureView) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sim encoder") });

        // Build a temporary bindgroup for update_grid requiring the depth map + mouse.
        let update_layout = self.pipelines.update_grid.get_bind_group_layout(0);
        let dynamic_update_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("updateGridBGFull"),
            layout: &update_layout,
            entries: &[
                // 0-3 mirror earlier partial layout
                wgpu::BindGroupEntry { binding: 0, resource: self.buffers.cell_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buffers.real_box_size.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.buffers.init_box_size.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.buffers.render_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(depth_map_view) },
                wgpu::BindGroupEntry { binding: 5, resource: self.buffers.mouse_uniform.as_entire_binding() },
            ],
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("mpm"), timestamp_writes: None });
            let particles = self.config.num_particles;
            let grids = self.grid_count;

            cpass.set_pipeline(&self.pipelines.clear_grid);
            cpass.set_bind_group(0, &self.bind_groups.clear_grid, &[]);
            cpass.dispatch_workgroups((grids + 63) / 64, 1, 1);

            cpass.set_pipeline(&self.pipelines.p2g1);
            cpass.set_bind_group(0, &self.bind_groups.p2g1, &[]);
            cpass.dispatch_workgroups((particles + 63) / 64, 1, 1);

            cpass.set_pipeline(&self.pipelines.p2g2);
            cpass.set_bind_group(0, &self.bind_groups.p2g2, &[]);
            cpass.dispatch_workgroups((particles + 63) / 64, 1, 1);

            cpass.set_pipeline(&self.pipelines.update_grid);
            cpass.set_bind_group(0, &dynamic_update_bg, &[]);
            cpass.dispatch_workgroups((grids + 63) / 64, 1, 1);

            cpass.set_pipeline(&self.pipelines.g2p);
            cpass.set_bind_group(0, &self.bind_groups.g2p, &[]);
            cpass.dispatch_workgroups((particles + 63) / 64, 1, 1);

            cpass.set_pipeline(&self.pipelines.copy_pos);
            cpass.set_bind_group(0, &self.bind_groups.copy_pos, &[]);
            cpass.dispatch_workgroups((particles + 63) / 64, 1, 1);
        }

        self.queue.submit([encoder.finish()]);
    }
}
