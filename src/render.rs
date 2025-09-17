//! Rendering module: creates surface-dependent resources and renders particles provided by Simulation.
use std::borrow::Cow;
use std::sync::Arc;

use wgpu::util::DeviceExt;
use winit::{dpi::PhysicalSize, window::Window};

use crate::{simulation::Simulation, types::{MouseInfoUniform, inject_overrides}};

const SPHERE_WGSL_RAW: &str = include_str!("../WaterBall/render/sphere.wgsl");

pub struct RenderTextures {
    pub depth_test_view: wgpu::TextureView,
    pub depth_map_view: wgpu::TextureView,
}

pub struct Renderer {
    window: Arc<Window>,
    pub surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    pub size: PhysicalSize<u32>,
    sphere_pipeline: wgpu::RenderPipeline,
    sphere_bind_group: wgpu::BindGroup,
    textures: RenderTextures,
}

impl Renderer {
    pub fn new(window: Arc<Window>, surface: wgpu::Surface<'static>, adapter: &wgpu::Adapter, sim: &Simulation, size: PhysicalSize<u32>) -> Result<Self, ()> {
        let caps = surface.get_capabilities(adapter);
        let surface_format = caps.formats.iter().copied().find(|f| f.is_srgb()).unwrap_or(caps.formats[0]);

        let textures = Self::create_textures(sim.device(), size);

        let sphere_pipeline; // needs shader module and layout with simulation buffers
        let sphere_bind_group;
        {
            let mut sphere_src = inject_overrides(SPHERE_WGSL_RAW, &[("restDensity", sim.config.rest_density), ("densitySizeScale", 4.0)]);
            sphere_src = sphere_src.replace(
                "let size = uniforms.sphere_size * clamp(particles[instance_index].density / restDensity * densitySizeScale, 0.0, 1.0);",
                "let size = uniforms.sphere_size;",
            );
            let sphere_mod = sim.device().create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("sphere_mod"),
                source: wgpu::ShaderSource::Wgsl(Cow::Owned(sphere_src)),
            });
            sphere_pipeline = sim.device().create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("sphere_pipeline"),
                layout: None,
                vertex: wgpu::VertexState { module: &sphere_mod, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
                fragment: Some(wgpu::FragmentState { module: &sphere_mod, entry_point: Some("fs"), compilation_options: Default::default(), targets: &[
                    Some(wgpu::ColorTargetState { format: surface_format, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::R32Float, blend: None, write_mask: wgpu::ColorWrites::RED }),
                ] }),
                primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
                depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });

            sphere_bind_group = sim.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sphereBG"),
                layout: &sphere_pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: sim.buffers.posvel_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: sim.buffers.render_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: sim.device().create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("stretchStrength"), contents: bytemuck::bytes_of(&2.0_f32), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST }).as_entire_binding() },
                ],
            });
        }

        let mut renderer = Self { window, surface, surface_format, size, sphere_pipeline, sphere_bind_group, textures };
        renderer.configure_surface(sim.device());
        Ok(renderer)
    }

    fn create_textures(device: &wgpu::Device, size: PhysicalSize<u32>) -> RenderTextures {
        let depth_test_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth32float"),
            size: wgpu::Extent3d { width: size.width.max(1), height: size.height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_map_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depthmap_r32float"),
            size: wgpu::Extent3d { width: size.width.max(1), height: size.height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        RenderTextures { depth_test_view: depth_test_tex.create_view(&Default::default()), depth_map_view: depth_map_tex.create_view(&Default::default()) }
    }

    fn configure_surface(&mut self, device: &wgpu::Device) {
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
        self.surface.configure(device, &config);
    }

    pub fn resize(&mut self, sim: &Simulation, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 { return; }
        self.size = new_size;
        self.configure_surface(sim.device());
        self.textures = Self::create_textures(sim.device(), new_size);
        sim.update_render_uniforms(new_size);
    }

    pub fn draw(&mut self, sim: &Simulation, mouse: MouseInfoUniform) {
        sim.update_mouse(mouse);
        // Acquire surface texture (reconfigure on errors)
        let surface_tex = match self.surface.get_current_texture() {
            Ok(st) => st,
            Err(_) => { self.configure_surface(sim.device()); self.surface.get_current_texture().expect("acquire surface") }
        };
        let surface_view = surface_tex.texture.create_view(&Default::default());

        // First run compute to update particle buffers (needs depth map view as earlier frame state; currently empties each frame)
        sim.step(&self.textures.depth_map_view);

        let mut encoder = sim.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("render encoder") });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sphere pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment { view: &surface_view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.03, a: 1.0 }), store: wgpu::StoreOp::Store } }),
                    Some(wgpu::RenderPassColorAttachment { view: &self.textures.depth_map_view, depth_slice: None, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 1e6, g: 0.0, b: 0.0, a: 1.0 }), store: wgpu::StoreOp::Store } }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &self.textures.depth_test_view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.sphere_pipeline);
            rpass.set_bind_group(0, &self.sphere_bind_group, &[]);
            rpass.draw(0..6, 0..sim.config.num_particles);
        }
        sim.queue().submit([encoder.finish()]);
        self.window.pre_present_notify();
        surface_tex.present();
    }
}
