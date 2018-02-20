# put here the root name of your code
PROGRAM	= regrid

CC	= gcc
CFLAGS	= -c -fPIC -O2  

# determines which python version the user has
pyv="$(python -c 'import sys; print(sys.version_info[0])')"
# takes action depending on the version
ifeq ($pyv,3)
	LFLAGS  = -I/Users/nemmen/anaconda3/include/python3.5m -I/Users/nemmen/anaconda3/lib/python3.5/site-packages/numpy/core/include
else 
	LFLAGS  = -I/Users/nemmen/anaconda3/envs/py27/include/python2.7 -I/Users/nemmen/anaconda3/envs/py27/lib/python2.7/site-packages/numpy/core/include
endif 

all: ${PROGRAM}

${PROGRAM}:	${PROGRAM}.c
	[ -f ./numpy.i ] && echo "numpy.i already here, good" || wget https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i

	swig -python -o ${PROGRAM}_wrap.c ${PROGRAM}.i
	$(CC) ${CFLAGS} ${PROGRAM}.c -o ${PROGRAM}.o
	$(CC) ${CFLAGS} ${PROGRAM}_wrap.c -o ${PROGRAM}_wrap.o ${LFLAGS}
	ld -bundle -flat_namespace -undefined suppress -o _${PROGRAM}.so *.o

clean:
	rm -rf *.o *.mod *.so ${PROGRAM}_wrap.c numpy.i __pycache__

