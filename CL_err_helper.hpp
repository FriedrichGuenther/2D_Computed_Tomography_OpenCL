#if defined(__APPLE__) 
#include <OpenCL/cl2.hpp>
#else 
#include <CL/cl2.hpp>
#endif

#define CLErrHelperSwitchHelper(x) case x: return #x;

char const * opencl_err_to_str(cl_int err)
{
    switch (err)
    {
        CLErrHelperSwitchHelper(CL_SUCCESS                                  )
        CLErrHelperSwitchHelper(CL_DEVICE_NOT_FOUND                         )
        CLErrHelperSwitchHelper(CL_DEVICE_NOT_AVAILABLE                     )
        CLErrHelperSwitchHelper(CL_COMPILER_NOT_AVAILABLE                   )
        CLErrHelperSwitchHelper(CL_MEM_OBJECT_ALLOCATION_FAILURE            )
        CLErrHelperSwitchHelper(CL_OUT_OF_RESOURCES                         )
        CLErrHelperSwitchHelper(CL_OUT_OF_HOST_MEMORY                       )
        CLErrHelperSwitchHelper(CL_PROFILING_INFO_NOT_AVAILABLE             )
        CLErrHelperSwitchHelper(CL_MEM_COPY_OVERLAP                         )
        CLErrHelperSwitchHelper(CL_IMAGE_FORMAT_MISMATCH                    )
        CLErrHelperSwitchHelper(CL_IMAGE_FORMAT_NOT_SUPPORTED               )
        CLErrHelperSwitchHelper(CL_BUILD_PROGRAM_FAILURE                    )
        CLErrHelperSwitchHelper(CL_MAP_FAILURE                              )
        #ifdef CL_VERSION_1_1
        CLErrHelperSwitchHelper(CL_MISALIGNED_SUB_BUFFER_OFFSET             )
        CLErrHelperSwitchHelper(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
        #endif
        #ifdef CL_VERSION_1_2
        CLErrHelperSwitchHelper(CL_COMPILE_PROGRAM_FAILURE                  )
        CLErrHelperSwitchHelper(CL_LINKER_NOT_AVAILABLE                     )
        CLErrHelperSwitchHelper(CL_LINK_PROGRAM_FAILURE                     )
        CLErrHelperSwitchHelper(CL_DEVICE_PARTITION_FAILED                  )
        CLErrHelperSwitchHelper(CL_KERNEL_ARG_INFO_NOT_AVAILABLE            )
        #endif

        CLErrHelperSwitchHelper(CL_INVALID_VALUE                            )
        CLErrHelperSwitchHelper(CL_INVALID_DEVICE_TYPE                      )
        CLErrHelperSwitchHelper(CL_INVALID_PLATFORM                         )
        CLErrHelperSwitchHelper(CL_INVALID_DEVICE                           )
        CLErrHelperSwitchHelper(CL_INVALID_CONTEXT                          )
        CLErrHelperSwitchHelper(CL_INVALID_QUEUE_PROPERTIES                 )
        CLErrHelperSwitchHelper(CL_INVALID_COMMAND_QUEUE                    )
        CLErrHelperSwitchHelper(CL_INVALID_HOST_PTR                         )
        CLErrHelperSwitchHelper(CL_INVALID_MEM_OBJECT                       )
        CLErrHelperSwitchHelper(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          )
        CLErrHelperSwitchHelper(CL_INVALID_IMAGE_SIZE                       )
        CLErrHelperSwitchHelper(CL_INVALID_SAMPLER                          )
        CLErrHelperSwitchHelper(CL_INVALID_BINARY                           )
        CLErrHelperSwitchHelper(CL_INVALID_BUILD_OPTIONS                    )
        CLErrHelperSwitchHelper(CL_INVALID_PROGRAM                          )
        CLErrHelperSwitchHelper(CL_INVALID_PROGRAM_EXECUTABLE               )
        CLErrHelperSwitchHelper(CL_INVALID_KERNEL_NAME                      )
        CLErrHelperSwitchHelper(CL_INVALID_KERNEL_DEFINITION                )
        CLErrHelperSwitchHelper(CL_INVALID_KERNEL                           )
        CLErrHelperSwitchHelper(CL_INVALID_ARG_INDEX                        )
        CLErrHelperSwitchHelper(CL_INVALID_ARG_VALUE                        )
        CLErrHelperSwitchHelper(CL_INVALID_ARG_SIZE                         )
        CLErrHelperSwitchHelper(CL_INVALID_KERNEL_ARGS                      )
        CLErrHelperSwitchHelper(CL_INVALID_WORK_DIMENSION                   )
        CLErrHelperSwitchHelper(CL_INVALID_WORK_GROUP_SIZE                  )
        CLErrHelperSwitchHelper(CL_INVALID_WORK_ITEM_SIZE                   )
        CLErrHelperSwitchHelper(CL_INVALID_GLOBAL_OFFSET                    )
        CLErrHelperSwitchHelper(CL_INVALID_EVENT_WAIT_LIST                  )
        CLErrHelperSwitchHelper(CL_INVALID_EVENT                            )
        CLErrHelperSwitchHelper(CL_INVALID_OPERATION                        )
        CLErrHelperSwitchHelper(CL_INVALID_GL_OBJECT                        )
        CLErrHelperSwitchHelper(CL_INVALID_BUFFER_SIZE                      )
        CLErrHelperSwitchHelper(CL_INVALID_MIP_LEVEL                        )
        CLErrHelperSwitchHelper(CL_INVALID_GLOBAL_WORK_SIZE                 )
        #ifdef CL_VERSION_1_1
        CLErrHelperSwitchHelper(CL_INVALID_PROPERTY                         )
        #endif
        #ifdef CL_VERSION_1_2
        CLErrHelperSwitchHelper(CL_INVALID_IMAGE_DESCRIPTOR                 )
        CLErrHelperSwitchHelper(CL_INVALID_COMPILER_OPTIONS                 )
        CLErrHelperSwitchHelper(CL_INVALID_LINKER_OPTIONS                   )
        CLErrHelperSwitchHelper(CL_INVALID_DEVICE_PARTITION_COUNT           )
        #endif
        #ifdef CL_VERSION_2_0
        CLErrHelperSwitchHelper(CL_INVALID_PIPE_SIZE                        )
        CLErrHelperSwitchHelper(CL_INVALID_DEVICE_QUEUE                     )
        #endif
        #ifdef CL_VERSION_2_2
        CLErrHelperSwitchHelper(CL_INVALID_SPEC_ID                          )
        CLErrHelperSwitchHelper(CL_MAX_SIZE_RESTRICTION_EXCEEDED            )
        #endif
        default: return "UNKNOWN!!!!";
    }
}


