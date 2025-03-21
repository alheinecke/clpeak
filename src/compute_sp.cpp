#include <clpeak.h>

int clPeak::runComputeSP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
  float timed, gflops;
  cl_uint workPerWI;
  cl::NDRange globalSize, localSize;
  cl_float A = 1.3f;
  cl_float B = 1.3f;
  uint iters = devInfo.computeIters;

  if (!isComputeSP)
    return 0;

  try
  {
    log->print(NEWLINE TAB TAB "Single-precision compute (GFLOPS)" NEWLINE);
    log->xmlOpenTag("single_precision_compute");
    log->xmlAppendAttribs("unit", "gflops");

    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();

    uint64_t globalWIs = (devInfo.numCUs) * (devInfo.computeWgsPerCU) * (devInfo.maxWGSize);
    uint64_t t = std::min((globalWIs * sizeof(cl_float)), devInfo.maxAllocSize) / sizeof(cl_float);
    globalWIs = roundToMultipleOf(t, devInfo.maxWGSize);

    cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (globalWIs * sizeof(cl_float)));

    globalSize = globalWIs;
    localSize = devInfo.maxWGSize;

    cl::Kernel kernel_v1(prog, "compute_sp_v1");
    kernel_v1.setArg(0, outputBuf), kernel_v1.setArg(1, A), kernel_v1.setArg(2, B);

    cl::Kernel kernel_v2(prog, "compute_sp_v2");
    kernel_v2.setArg(0, outputBuf), kernel_v2.setArg(1, A), kernel_v2.setArg(2, B);

    cl::Kernel kernel_v4(prog, "compute_sp_v4");
    kernel_v4.setArg(0, outputBuf), kernel_v4.setArg(1, A), kernel_v4.setArg(2, B);

    cl::Kernel kernel_v8(prog, "compute_sp_v8");
    kernel_v8.setArg(0, outputBuf), kernel_v8.setArg(1, A), kernel_v8.setArg(2, B);

    cl::Kernel kernel_v16(prog, "compute_sp_v16");
    kernel_v16.setArg(0, outputBuf), kernel_v16.setArg(1, A), kernel_v16.setArg(2, B);

    ///////////////////////////////////////////////////////////////////////////
    // Vector width 1
    if (!forceTest || strcmp(specifiedTestName, "float") == 0)
    {
      log->print(TAB TAB TAB "float   : ");

      workPerWI = 4096; // Indicates flops executed per work-item

      timed = run_kernel(queue, kernel_v1, globalSize, localSize, iters);

      gflops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;

      log->print(gflops);
      log->print(NEWLINE);
      log->xmlRecord("float", gflops);
    }
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 2
    if (!forceTest || strcmp(specifiedTestName, "float2") == 0)
    {
      log->print(TAB TAB TAB "float2  : ");

      workPerWI = 4096;

      timed = run_kernel(queue, kernel_v2, globalSize, localSize, iters);

      gflops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;

      log->print(gflops);
      log->print(NEWLINE);
      log->xmlRecord("float2", gflops);
    }
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 4
    if (!forceTest || strcmp(specifiedTestName, "float4") == 0)
    {
      log->print(TAB TAB TAB "float4  : ");

      workPerWI = 4096;

      timed = run_kernel(queue, kernel_v4, globalSize, localSize, iters);

      gflops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;

      log->print(gflops);
      log->print(NEWLINE);
      log->xmlRecord("float4", gflops);
    }
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 8
    if (!forceTest || strcmp(specifiedTestName, "float8") == 0)
    {
      log->print(TAB TAB TAB "float8  : ");

      workPerWI = 4096;

      timed = run_kernel(queue, kernel_v8, globalSize, localSize, iters);

      gflops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;

      log->print(gflops);
      log->print(NEWLINE);
      log->xmlRecord("float8", gflops);
    }
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 16
    if (!forceTest || strcmp(specifiedTestName, "float16") == 0)
    {
      log->print(TAB TAB TAB "float16 : ");

      workPerWI = 4096;

      timed = run_kernel(queue, kernel_v16, globalSize, localSize, iters);

      gflops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;

      log->print(gflops);
      log->print(NEWLINE);
      log->xmlRecord("float16", gflops);
    }
    ///////////////////////////////////////////////////////////////////////////
    log->xmlCloseTag(); // single_precision_compute
  }
  catch (cl::Error &error)
  {
    stringstream ss;
    ss << error.what() << " (" << error.err() << ")" NEWLINE
       << TAB TAB TAB "Tests skipped" NEWLINE;
    log->print(ss.str());
    return -1;
  }

  return 0;
}
