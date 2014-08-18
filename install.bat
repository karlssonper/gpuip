pushd %~dp0
if not exist build\ (
  echo "Run 'build.bat' first to build gpuip"
  
) else (
  cd build
  cmake --build . --config Release --target INSTALL	
)
pause
popd