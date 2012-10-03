This is a Matlab interface to the Huygens On Speed simulator.

Build mex-file using buildMex.m
 - null arguments builds mex in release mode
 - true as argument builds mex in debug mode
 - mex-file is linked with lib-files in the folder cuda_X_Y (specified with -L during compilation)
      - Before compiling make sure HuygensOnSpeed.lib is located in this folder
      - cduartxx_xx_x.dll also needs to be located here


IMPORTANT:

If your GPU is used for both display as well as for CUDA the simulator will fail if your kernel executes for too long.
In windows, you can change this value in the registry editor: (WIN-key + R, and then type 'regedit' and press enter)
HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\ GraphicsDrivers --> TdrLevel = <yourTimeoutValueInSeconds>
However, this is NOT RECOMMENDED!!!! So, proceed at your own risk!!! >.<
Another possible solution (though a bit costly :)) is to buy another GPU and use it in compute-mode only.