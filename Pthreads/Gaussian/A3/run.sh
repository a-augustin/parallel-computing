for size in 512 1024 2048 4096
   do
	for threads in 4 8 16 32
	do
            echo gauss $size $threads
	   ./gauss_eliminate $size $threads
        done
  done
