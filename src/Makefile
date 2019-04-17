# put here the root name of your code
PROGRAM	= fastregrid

CC	= pgcc
CFLAGS	= -acc -fPIC -O2 -Minfo=accel -c

# determines which python version the user has
PYV := $(shell python -c 'import sys; print(sys.version_info[0])')
# python minor version
PYV_MIN := $(shell python -c 'import sys; print(sys.version_info[1])')
# which OS do we have?
OS := $(shell uname)

# if the user is running MacOS
ifeq ($(OS), Darwin)
	# takes action depending on the python version
	ifeq ($(PYV),3)
		LFLAGS  = -I$(HOME)/anaconda3/include/python3.$(PYV_MIN)m -I$(HOME)/anaconda3/lib/python3.$(PYV_MIN)/site-packages/numpy/core/include 
		LDFLAGS= -bundle -flat_namespace -undefined suppress
	else 
		# improve: find the path below automatically
		LFLAGS  = -I$(HOME)/anaconda2/envs/py27/include/python2.7 -I$(HOME)/anaconda2/envs/py27/lib/python2.7/site-packages/numpy/core/include
		LDFLAGS= -bundle -flat_namespace -undefined suppress
	endif 
endif
# if the user is running Linux
ifeq ($(OS), Linux)
	ifeq ($(PYV),3)
		LFLAGS  = -I$(HOME)/anaconda3/include/python3.$(PYV_MIN)m -I$(HOME)/anaconda3/lib/python3.$(PYV_MIN)/site-packages/numpy/core/include
		LDFLAGS= -shared -undefined suppress
	else 
		LFLAGS  = -I$(HOME)/anaconda2/include/python2.7 -I$(HOME)/anaconda2/lib/python2.7/site-packages/numpy/core/include
		LDFLAGS= -shared -undefined suppress
	endif 
endif



all: ${PROGRAM}

${PROGRAM}:	${PROGRAM}.c
	[ -f ./numpy.i ] && echo "numpy.i already here, good" || wget https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i

	swig -python -o ${PROGRAM}_wrap.c ${PROGRAM}.i
	$(CC) ${CFLAGS} ${PROGRAM}.c -o ${PROGRAM}.o
	$(CC) ${CFLAGS} ${PROGRAM}_wrap.c -o ${PROGRAM}_wrap.o ${LFLAGS}
	ld ${LDFLAGS} -o _${PROGRAM}.so *.o

clean:
	rm -rf *.o *.mod *.so ${PROGRAM}_wrap.c numpy.i __pycache__ ${PROGRAM}.py *.pyc

