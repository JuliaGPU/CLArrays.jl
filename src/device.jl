import GPUArrays: is_gpu, is_cpu, name, threads, blocks, global_memory, local_memory

function devices()
    filter(cl.devices()) do dev
        # These drivers have some problems -  TODO figure out if we are the problem
        !(
            contains(cl.info(dev, :version), "AMD-APP (2348.3)") || #AMD for intel. Why though?
            contains(cl.info(dev, :version), "(Build 10)") # intels experimental 2.1 driver
        )
    end
end

is_gpu(dev::cl.Device) = cl.info(dev, :device_type) == :gpu
is_cpu(dev::cl.Device) = cl.info(dev, :device_type) == :cpu

name(dev::cl.Device) = string("CL ", cl.info(dev, :name))
function cl_version(dev::cl.Device)
    ver_str = cl.info(dev, :version)
    vmatch = match(r"\d.\d", ver_str)
    major, minor = parse.(Int, split(vmatch.match, '.'))
    VersionNumber(major, minor)
end

threads(dev::cl.Device) = cl.info(dev, :max_work_group_size) |> Int
blocks(dev::cl.Device) = cl.info(dev, :max_work_item_size)

global_memory(dev::cl.Device) = cl.info(dev, :global_mem_size) |> Int
local_memory(dev::cl.Device) = cl.info(dev, :local_mem_size) |> Int
