function buildMex(debug)

if exist('HuygensOnSpeed.mexw64', 'file')
    delete HuygensOnSpeed.mexw64
end

if nargin < 1
    debug = false;
end

if debug
    mex -v -g -win64 COMPFLAGS="$COMPFLAGS /MT" -I"../HuygensOnSpeed" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.0\include" -lHuygensOnSpeedD -L"cuda_4_0" HuygensOnSpeed.cpp
else
    %mex -v -win64 COMPFLAGS="$COMPFLAGS /MT /openmp" -I"../HuygensOnSpeed" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.0\include" -lHuygensOnSpeed -L"cuda_4_0" HuygensOnSpeed.cpp
    mex -v -win64 COMPFLAGS="$COMPFLAGS /MT /openmp" -I"../HuygensOnSpeed" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v4.2\include" -lHuygensOnSpeed -L"cuda_4_2" HuygensOnSpeed.cpp
end