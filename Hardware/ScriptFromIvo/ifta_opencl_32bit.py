import pyopencl as cl
import numpy as np
from PIL import Image
import time

"""
IFTA algorithm on OpenCL. Runs much faster than on CPU single thread.

Future improvements:
    * 2D Kernels with global index 1 as row and index 2 as column
    * Complex kernels to not have to invoke np.exp in the end.
"""

# The C-code used for the Kernel. There are three slightly different exponents
# that we calculate for GSW. Each has its own kernel. For now I map the 2D
# Image to a 1D array that I loop over. It might be smarter to keep 2
# dimensions and the GPU cores support up to 3. Spots could be a third.
knl_str = """
__kernel
void RS_single_spot_1d(__global uint *pixel, __global float *randoms,
                       __global float *output, 
                       uint width, uint height, float xm, float ym, 
                       float2 pref) {
    uint i = get_global_id(0);
    uint xj = pixel[i] % width;
    uint yj = pixel[i] / width;

    output[i] = pref.x * ((xj - width/2)*(xj - width/2) + (yj - height/2)*(yj - height/2)) + pref.y * (xj * xm + yj * ym + randoms[i]);
}

__kernel
void single_spot_1d(__global uint *pixel, __global float *output, 
                    uint width, uint height, float xm, float ym,
                    float2 pref) {
    uint i = get_global_id(0);
    uint xj = pixel[i] % width;
    uint yj = pixel[i] / width;

    output[i] = pref.x * ((xj - width/2)*(xj - width/2) + (yj - height/2)*(yj - height/2)) + pref.y * (xj * xm + yj * ym);
}

__kernel
void Vm_single_spot_1d(__global uint *pixel, __global float *phi, __global float *output, 
                    uint width, uint height, float xm, float ym,
                    float2 pref) {
    uint i = get_global_id(0);

    uint xj = pixel[i] % width;
    uint yj = pixel[i] / width;

    //printf("%u %f %f %f %u %u %f \\n", i, phi[i], xm, ym, *&xj, *&yj, phi[i] - pref.y * (xj * xm + yj * ym));
    output[i] = phi[i] - pref.x * ((xj - width/2)*(xj - width/2) + (yj - height/2)*(yj - height/2)) - pref.y * (xj * xm + yj * ym);
}
"""
# To debug one can use print(platform.extensions) to see on which device the
# required opencl extension (cl_khr_f64 I believe) is present.
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]  # This is our GPU.
ctx = cl.Context([device])
#ctx = cl.create_some_context()

# Compile the kernel code and prepare python objects.
program = cl.Program(ctx, knl_str).build()
knl_RS = program.RS_single_spot_1d
knl_GSW = program.single_spot_1d
knl_Vm = program.Vm_single_spot_1d

# Declare the pattern that we want.
lamb = 820-9
f = 4e-3
height = 1200 
width = 1920
pixel_size = 4e-3/height
spots = np.array([], dtype=np.float32)

n_x = 3
n_y = 3
n_z = 1
spot_distance = 25
start_x = -(n_x-1)/2 * spot_distance
start_y = -(n_y-1)/2 * spot_distance
start_z = -(n_z-1)/2 * spot_distance
for i in range(n_x):
    for j in range(n_y):
        for k in range(n_z):
            if spots.size == 0:
                spots = np.array([start_x + i*spot_distance,
                                  start_y + j*spot_distance,
                                  start_z + k*spot_distance], dtype=np.float32)
            else:
                spots = np.vstack([spots, [start_x + i*spot_distance,
                                           start_y + j*spot_distance,
                                           start_z + k*spot_distance]])

# Initialize the arrays with the correct data types.
pixels = np.arange(width*height, dtype=np.uint32)
phi_out = np.empty(pixels.shape, dtype=np.float32)
prefactors = np.array([np.pi/height**2, 2*np.pi/height],
                      dtype=np.float32)

start = time.time()
# Create the buffers that will be used during all the calculations.
mf = cl.mem_flags
px_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pixels)

# First get an estimated phase mask with SR algorithm.
exp_buffer = np.empty(pixels.shape, dtype=np.complex64)
for k in range(spots.shape[0]):
    queue = cl.CommandQueue(ctx)
    dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, phi_out.nbytes)
    randoms = (np.random.rand(height * width) * np.pi * 2. - np.pi).astype(np.float32)
    rand_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=randoms)

    knl_RS.set_args(px_buf, rand_buf, dest_buf, 
           np.uint32(width), np.uint32(height), 
           np.float32(spots[k][0]), np.float32(spots[k][1]), 
           np.array([np.float32(prefactors[0]*spots[k][2]), np.float32(prefactors[1])]))
    cl.enqueue_nd_range_kernel(queue, knl_RS, (width*height, 1), None, allow_empty_ndrange=True)
   
    cl.enqueue_barrier(queue)  # wait till kernel is finished.

    # Read out from the dest_buffer
    output = np.empty_like(randoms)
    cl.enqueue_copy(queue, output, dest_buf)
    exp_buffer += np.exp(1j*(output))

    # Clear the buffers. This means reset the destination and remove the others
    cl.enqueue_fill_buffer(queue, dest_buf, np.float32(0), 0,
                           phi_out.nbytes).wait()
    rand_buf.release()
    queue.finish()
phi_out = np.angle(exp_buffer)


print("Phi after RS \n", phi_out)

# Next we calculate the contribution to each desired spot using Vm.
Vm = np.empty((spots.shape[0]), dtype=np.complex64)
for k in range(spots.shape[0]):
    queue = cl.CommandQueue(ctx)
    dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, phi_out.nbytes)
    phi_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=phi_out)
    knl_Vm.set_args(px_buf, phi_buf, dest_buf, 
            np.uint32(width), np.uint32(height),
            np.float32(spots[k][0]), np.float32(spots[k][1]),
            np.array([np.float32(prefactors[0]*spots[k][2]), np.float32(prefactors[1])]))
    cl.enqueue_nd_range_kernel(queue, knl_Vm, (width*height, 1), None, allow_empty_ndrange=True)
    cl.enqueue_barrier(queue)

    output = np.empty_like(phi_out)
    cl.enqueue_copy(queue, output, dest_buf)

    Vm[k] = np.sum(np.exp(1j*(output))) / (width * height)
    print(Vm[k])

    cl.enqueue_fill_buffer(queue, dest_buf, np.float32(0), 0,
                               phi_out.nbytes).wait()
    #knl_Vm = knl_Vm.from_int_ptr(1, True)
    queue.finish()
Ik = np.abs(Vm)**2
print(Ik)
e = np.sum(Ik)
u = 1 - (np.amax(Ik) - np.amin(Ik))/(np.amax(Ik) + np.amin(Ik))
print("Obtained starting pattern. \n e = {}, u = {} \n".format(e, u))

i = 0
e_limit = 0.9
u_limit = 0.999999
max_iter = 50
w = np.ones((spots.shape[0]), dtype=np.float32)
# Now we start the loop to optimize the phase mask.
while (e < e_limit or u < u_limit) and i < max_iter:
    w *= np.mean(np.abs(Vm))/np.abs(Vm)
    #print(w)

    exp_buffer = np.empty(shape=pixels.shape[0], dtype=np.complex64)
    for k in range(spots.shape[0]):
        queue = cl.CommandQueue(ctx)
        dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, phi_out.nbytes)
        knl_GSW.set_args(px_buf, dest_buf,
                np.uint32(width), np.uint32(height),
                np.float32(spots[k][0]), np.float32(spots[k][1]),
                np.array([np.float32(prefactors[0]*spots[k][2]), np.float32(prefactors[1])]))
        cl.enqueue_nd_range_kernel(queue, knl_GSW, (width*height, 1), None, allow_empty_ndrange=True)

        output = np.empty_like(phi_out)
        cl.enqueue_copy(queue, output, dest_buf)
        exp_buffer += np.exp(1j*output)*w[k]*Vm[k]/np.abs(Vm[k])

        cl.enqueue_fill_buffer(queue, dest_buf, np.float32(0), 0,
                               phi_out.nbytes).wait()
        queue.finish()

    phi_out = np.angle(exp_buffer)

    #phi_out = (np.random.rand(height * width) * np.pi * 2. - np.pi).astype(np.float32)
    phi_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=phi_out)

    Vm = np.empty((spots.shape[0]), dtype=np.complex64)
    for k in range(spots.shape[0]):
        queue = cl.CommandQueue(ctx)
        dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, phi_out.nbytes)
        phi_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=phi_out)
        knl_Vm.set_args(px_buf, phi_buf, dest_buf, 
                np.uint32(width), np.uint32(height),
                np.float32(spots[k][0]), np.float32(spots[k][1]),
                np.array([np.float32(prefactors[0]*spots[k][2]), np.float32(prefactors[1])]))
        cl.enqueue_nd_range_kernel(queue, knl_Vm, (width*height, 1), None, allow_empty_ndrange=True)

        output = np.empty_like(phi_out)
        cl.enqueue_copy(queue, output, dest_buf)

        Vm[k] = np.sum(np.exp(1j*(output))) / (width * height)


        cl.enqueue_fill_buffer(queue, dest_buf, np.float32(0), 0,
                                   phi_out.nbytes).wait()
        queue.finish()

    Ik = np.abs(Vm)**2
    print(Ik)
    e = np.sum(Ik)
    u = 1 - (np.amax(Ik) - np.amin(Ik))/(np.amax(Ik) + np.amin(Ik))
    i += 1
    print("Iteration {} finished. \n e = {}, u = {} \n".format(i, e, u))

    #if i%200 == 0:
    #    phase = (np.round(255 * (phi_out+np.pi) /
    #                      (2 * np.pi))).astype(np.uint8).reshape((height, width))
    #    im1 = Image.fromarray(phase)
    #    im1.save("pm_opencl_7x7_uniform_spots.bmp")

end = time.time()
print("Finished iterations. Total time {}".format(end-start))
phase = (np.round(255 * (phi_out+np.pi) /
                      (2 * np.pi))).astype(np.uint8).reshape((height, width))
#%%
im1 = Image.fromarray(phase)
im1.show()
im1.save("S:\KAT1\SLM\image files\from Ivo script opencl made lab pc/pm_opencl_4x4_spots.bmp")

phi = phase/(255)*2*np.pi-np.pi
pattern_8bit = np.abs(np.fft.fftshift(np.fft.fftn(np.exp(1j*phi),
                                                  norm="ortho")))**2
im2 = Image.fromarray((pattern_8bit /
                       np.amax(pattern_8bit)*255).astype(np.uint8))
im2.show()
