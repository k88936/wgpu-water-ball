mod types;
mod simulation;
mod render;

use std::sync::Arc;
use winit::{application::ApplicationHandler, event::WindowEvent, event_loop::{ActiveEventLoop, ControlFlow, EventLoop}, window::{Window, WindowId}};
use types::{MouseInfoUniform, SimConfig};
use simulation::Simulation;
use render::Renderer;

struct App {
    window: Option<Arc<Window>>,
    simulation: Option<Simulation>,
    renderer: Option<Renderer>,
}

impl Default for App { fn default() -> Self { Self { window: None, simulation: None, renderer: None } } }

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(event_loop.create_window(Window::default_attributes()).expect("window"));

    // Create instance, surface, then request adapter compatible with surface.
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let surface = instance.create_surface(window.clone()).expect("surface");
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, compatible_surface: Some(&surface), force_fallback_adapter: false })).expect("adapter");

        let size = window.inner_size();
    let sim = pollster::block_on(Simulation::new(&adapter, SimConfig::default(), size)).expect("simulation");
    let renderer = Renderer::new(window.clone(), surface, &adapter, &sim, size).expect("renderer");

        self.window = Some(window.clone());
        self.simulation = Some(sim);
        self.renderer = Some(renderer);
        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let (window, sim, renderer) = match (&self.window, &self.simulation, &mut self.renderer) {
            (Some(w), Some(s), Some(r)) => (w.clone(), s, r),
            _ => return,
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                renderer.resize(sim, size);
            }
            WindowEvent::RedrawRequested => {
                let size = renderer.size;
                let mouse = MouseInfoUniform { screen_size: [size.width as f32, size.height as f32], mouse_coord: [0.5, 0.5], mouse_vel: [0.0, 0.0], mouse_radius: 6.0, _pad: 0.0 };
                renderer.draw(sim, mouse);
                window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("run app");
}
