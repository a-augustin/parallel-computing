for size in 512 1024 2048 4096
   do
	for threads in 4 8 16 32
	do
        echo JACOBI: $size $threads
	   ./jacobi_solver $size $threads
        done
  done
