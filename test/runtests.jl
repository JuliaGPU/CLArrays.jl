using CLArrays
using GPUArrays.TestSuite, Base.Test
using GPUArrays: global_size
using CUDAnative, CUDAdrv
TestSuite.run_tests(CLArray)

using CLArrays

x = CLArray(rand(Float32, 10))

GPUArrays.gpu_call(x, (x,)) do state, l
    l[1] = 1f0 ^ 1.0
    return
end


using ArrayFire

ArrayFire.set_backend(AF_BACKEND_OPENCL)

using CLArrays
TY=Float32
N = 2^9
h    = TY(2*π/N)
epsn = TY(h * .5)
C    = TY(2/epsn)
tau  = TY(epsn * h)
Tfinal = 50.

S(x,y) = exp(-x^2/0.1f0)*exp(-y^2/0.1f0)

# real-space and reciprocal-space grids
# the real-space grid is just used for plotting!
X_cpu = convert.(TY,collect(linspace(-pi+h, pi, N)) .* ones(1,N))
X = CLArray(X_cpu);
k = collect([0:N/2; -N/2+1:-1]);
Â = CLArray(convert.(TY,kron(k.^2, ones(1,N)) + kron(ones(N), k'.^2)));

# initial condition
uc = TY(2.0)*(CLArray{TY}(rand(TY, (N, N)))-TY(0.5));

#################################################################
#################################################################
function take_step!(u, t_plot)
    uff = fft(u)
    uff1 = fft(u.^3f0-u)
    @. tmp1 = ((1f0+C*tau*Â) .* uff .- tau/epsn * (Â .* uff1)) ./ (1f0+(epsn*tau)*Â.^2f0+C*tau*Â)
	u .= real.(ifft(tmp1))
	nothing
end
function normalise_af!(u,out)
	out .= u-minimum(u)
	out .= out / maximum(out)
	nothing
end
#################################################################
#################################################################

T_plot = 0.01; t_plot = 0.0


up = copy(uc)
@time for n = 1:ceil(Tfinal / tau)
	@show n
  	take_step!(uc, t_plot)
	t_plot += tau
  # f t_plot >= T_plot
  # # @show n
  # normalise_af!(uc,up)
  # draw_image(window_, up, Ref(ArrayFire.af_cell(0, 0, pointer(""), ArrayFire.AF_COLORMAP_HEAT)))
  # set_title(window_,"N = $N, n = $n, t = $(round(n*tau,4)), max = $(maximum(uc)) - min = $(minimum(uc))")
  # # set_axes_limits_3d(window_,-3.2f0,3.2f0,-3.2f0,-3.2f0,-2.3f0,2.0f0,true,Ref(ArrayFire.af_cell(0, 0, pointer(""), ArrayFire.AF_COLORMAP_HEAT)))
  # show(window_)
  # t_plot = 0.0
  # end
end

print_mem_info("GPU-Mem apres loop", 0)
