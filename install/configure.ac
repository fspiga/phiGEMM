# Copyright (C) 2011-2012 Quantum ESPRESSO Foundation
# Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present package,
# or http://www.gnu.org/copyleft/gpl.txt .
#
# Author: Filippo Spiga (spiga.filippo@gmail.com)

AC_INIT(phiGEMM, 2.0, , phigemm)

AC_ARG_ENABLE(cpu_multithread,
   [AS_HELP_STRING([--enable-cpu-multithread],
       [enable multi-threading BLAS library for CPU (default: yes)])],
   [if   test "$enableval" = "yes" ; then
      use_multithread=1
   else
      use_multithread=0
   fi],
   [use_multithread=1])

AC_ARG_ENABLE(cublas,
   [AS_HELP_STRING([--enable-cublas],
       [phiGEMM uses CUBLAS for GPU GEMM operations (default: yes)])],
   [if   test "$enableval" = "yes" ; then
      use_cublas=1
   else
      use_cublas=0
   fi],
   [use_cublas=1])
   
AC_ARG_ENABLE(magmablas,
   [AS_HELP_STRING([--enable-magmablas],
       [phiGEMM uses MAGMABLAS for GPU GEMM operations (default: no)])],
   [if   test "$enableval" = "yes" ; then
      use_magmablas=1
   else
      use_magmablas=0
   fi],
   [use_magmablas=0])
   
AC_ARG_ENABLE(parallel,
   [AS_HELP_STRING([--enable-parallel],
       [enable the MPI support in case of self-init (default: no)])],
   [if   test "$enableval" = "yes" ; then
      use_parallel=1
   else
      use_parallel=0
   fi],
   [use_parallel=0])
   
AC_ARG_ENABLE(debug,
   [AS_HELP_STRING([--enable-debug],
       [compile Fortran with debug flags (default: no)])],
   [if   test "$enableval" = "yes" ; then
      use_debug=1
   else
      use_debug=0
   fi],
   [use_debug=0])
   
 AC_ARG_ENABLE(profiling,
   [AS_HELP_STRING([--enable-profiling],
       [enable the profiling of phiGEMM calls (default: no)])],
   [if   test "$enableval" = "yes" ; then
      use_profile=1
   else
      use_profile=0
   fi],
   [use_profile=0])

AC_ARG_ENABLE(multi_gpu, 
   [AS_HELP_STRING([--enable-multi-gpu],
       [enable multi GPU support (default: no)])],
   [if  test "$enableval" = "yes" ; then
      use_multigpu=1
   else
      use_multigpu=0
   fi],
   [use_multigpu=0])
   
AC_ARG_ENABLE(self_tune, 
   [AS_HELP_STRING([--enable-self-tune],
       [enable self-tune (default: no, experimental)])],
   [if  test "$enableval" = "yes" ; then
      use_selftune=1
   else
      use_selftune=0
   fi],
   [use_selftune=0])
      
AC_ARG_WITH(blas_dir, 
   [AS_HELP_STRING([--with-blas-dir=<path>],
    [specify BLAS library installation directory (default is LD_LIBRARY_PATH)])],
    [blas_lib_path="$withval"], 
    [blas_lib_path="/usr/lib64"])
    
AC_ARG_WITH(cuda_dir, 
   [AS_HELP_STRING([--with-cuda-dir=<path>],
    [specify CUDA installation directory (default is /usr/local/cuda/)])],
    [cuda_path="$withval"], 
    [cuda_path="/usr/local/cuda/"])

AC_ARG_WITH(magma_dir, 
   [AS_HELP_STRING([--with-magma-dir=<path>],
   	[specify MAGMA installation directory (default is /usr/local/magma/)])],
    [magma_path="$withval"], 
    [magma_path="/usr/local/magma/"])

AC_ARG_WITH(cpu_only,
   [AS_HELP_STRING([--with-cpu-only],
       [everything is performed only by the CPU, useful for call-by-call profiling (default: no)])],
    [if   test "$withval" = "yes" ; then
      use_cpu_only=1
   else
      use_cpu_only=0
   fi],
   [use_cpu_only=0])

AC_ARG_WITH(gpu_only,
   [AS_HELP_STRING([--with-gpu-only],
       [everything is performed only by the GPU, useful for call-by-call profiling (default: no)])],
    [if   test "$withval" = "yes" ; then
      use_gpu_only=1
   else
      use_gpu_only=0
   fi],
   [use_gpu_only=0])
   
AC_ARG_WITH(pinned, 
   [AS_HELP_STRING([--with-mem-pinned],
       [assume host memory is PINNED (default: no)])],
   [if  test "$enableval" = "yes" ; then
      use_pinned=1
   else
      use_pinned=0
   fi],
   [use_pinned=0])
   
AC_ARG_WITH(split_by_hand,
   [AS_HELP_STRING([--with-split-by-hand],
       [the split factor has to be specified in the code instead of using env variables (default: no)])],
    [if   test "$withval" = "yes" ; then
      use_explicit_factor=1
   else
      use_explicit_factor=0
   fi],
   [use_explicit_factor=0])
   
AC_ARG_WITH(special_k,
   [AS_HELP_STRING([--with-special-k],
       [enable SPECIAL-K for bad-shaped rectangular matrices (default: no)])],
    [if   test "$withval" = "yes" ; then
      use_special_k=1
   else
      use_special_k=0
   fi],
   [use_special_k=0])
   
   
# -----------------------------
# store variables from the environment, if set (may be or not be set)
# If set, they take precedence over configure internal choice.
# Flags and libraries are accepted without further testing;
# compilers are tested. Specify compiler name only, not the full path
# (i.e. F90=/usr/local/bin/f90 may not work, use F90=f90)
topdir=$TOPDIR			# current directory
arch=$ARCH				# see below for recognized architectures
env_cc=$CC				# C compiler (must be in the execution path)
cflags=$CFLAGS			# Flags for C compiler
cppflags=$CPPFLAGS		# Flags for C preprocessor
ld=$LD					# Loader (must be in the execution path)
ldflags=$LDFLAGS		# Flags for loader
ld_libs=$LD_LIBS		# Additional libraries
blas_libs=$BLAS_LIBS	# blas library
libdirs=$LIBDIRS		# Where to look for libraries (e.g. /my/blas/lib)
ar=$AR					# ar (shouldn't be needed)
arflags=$ARFLAGS		# Flags for ar (as above)
nvcc=$NVCC				# NVIDIA CUDA compiler 
nvccflags=$NVCCFLAGS	# Flags for NVIDIA CUDA compiler
cuda_path=$CUDA_HOME	# CUDA lib
magma_path=$MAGMA_HOME	# MAGMA lib

# configure for current directory by default
if test "$topdir" = "" ; then topdir="`pwd`" ; fi

# check system type (no cross-compilation for now)
AC_CANONICAL_BUILD

# From Quantum ESPRESSO configure:
# many HPC systems are configured so that running parallel programs
# interactively is disabled: on those systems, AC_PROG_FC / _F77 / _CC
# would fail because they can't run the compiled executables.
# to work around that, let's pretend we are cross-compiling even if we aren't
# !!! this relies on undocumented Autoconf behavior !!!

#cross_compiling=yes

# This is used to distinguish between true and fake cross compilation
# (only on NEC SX8 actually)
# if test "$host" != "" ; then ranlib=echo; fi

if test "$host" = "" ; then host=$build; fi

# identify architecture
if test "$arch" = ""
then
        case $host in
                ia64-*-linux-gnu )      arch=ia64   ;;
                x86_64-*-linux-gnu )    arch=x86_64 ;;
                *-pc-linux-gnu )        arch=ia32   ;;
                *-ibm-aix* )            arch=aix    ;;
                sparc-sun-* )           arch=sparc  ;;
                i386-pc-solaris* )      arch=solaris;;
                powerpc-apple-darwin* ) arch=macppc ;;
                i686-apple-darwin* )    arch=mac686 ;;
                *-pc-cygwin )           arch=cygwin ;;
                sx*-nec* )              arch=necsx  ;;
                powerpc64-*-linux-gnu ) arch=ppc64  ;;
                * )                     AC_MSG_WARN(incorrect host name ?)
        ;;
        esac
    # workarounf for Cray-XT machines
        test -d /proc/cray_xt && arch=crayxt
    # workarounf for IBM BG machines
        test -d /bgsys && arch=ppc64-bg
fi
echo checking architecture... $arch

# candidate archiver and archiver flags
try_ar="ar"
try_arflags="ruv"

# Preprocessor
cpp="cpp"
cppflags="-P -traditional"

# check serial Fortran 90 compiler. This must be done before performing
# the check for the parallel compiler (section below) because option
# --disable-parallel will do very strange things otherwise. The reason
# seems to be that autoconf does not repeat all tests for the second
# occurrence of AC_PROG_FC. So the first occurrence is the one that
# must always be performed, the second is optional. PG & CC sep.2006

# INTEL - PGI - GNU - CRAY WRAPPER
try_f90="ifort pgf90 gfortran ftn"
try_cc="icc pgcc gcc cc"

# use F90 if set
if test "$f90" = "" ; then f90="$try_f90" ; fi
AC_PROG_FC($f90)
f90=$FC

echo setting F90... $f90

case "$f90" in
f90 | fc | ftn )
    echo $ECHO_N "checking version wrapped by $f90 command... $ECHO_C"

    if $f90 -V 2>&1 | grep -q "Intel(R)" ; then
        f90_version=ifort
    elif $f90 -V 2>&1 | grep -q "^pgf" ; then
        f90_version=pgf
    elif $f90 -v 2>&1 | grep -q "gcc version" ; then
        f90_version=gfortran
    else
        echo $ECHO_N "unknown, leaving as... $ECHO_C"
        f90_version=$f90
    fi
    echo $f90_version
    ;;
* )
    f90_version=$f90
    ;;
esac

case "$arch:$f90_version" in
*:ifort* )
    try_cc="icc ecc $try_cc"
    ;;
*:pgf90 )
    try_cc="pgcc $try_cc"
    ;;
cray*:* )
    try_cc="cc"
    ;;
* )
    #ABORT
    ;;
esac


# check serial C compiler
if test "$env_cc" = "" ; then cc="$try_cc" ; else cc="$env_cc"; fi
AC_PROG_CC($cc)
cc=$CC

echo setting CC... $cc

# No MACRO by defaults...
try_phigemm_opts=""

case "$arch:$f90_version" in
ia32:ifort* | ia64:ifort* | x86_64:ifort*  )
	try_ccflags="-O3 -fPIC"
	try_ldflags="-O3 -fPIC"
	try_ldsharedflag="--shared"
	pre_fdflags="-fpp"
	try_f90flags="-module ../include/"
    if test "$use_debug" -eq 1; then
        try_fflags="$try_fflags -g -traceback -par-report0 -vec-report0"
    fi
    try_openmp="-openmp"
    try_phigemm_opts="-D__INTEL"
    ;;
ia32:pgf* | ia64:pgf* | x86_64:pgf* )
	try_ccflags="-O3 -fPIC"
	try_ldflags="-O3 -fPIC"
	try_ldsharedflag="--shared"
	pre_fdflags="-Mpreprocess"
	try_f90flags="-module ../include/"
    try_openmp="-mp"
    try_phigemm_opts="-D__PGI"
    have_cpp=0 #???
	;;
*:gfortran )
	try_ccflags="-O3 -fPIC"
	try_ldflags="-O3 -fPIC"
	try_ldsharedflag="-shared" # to check
	pre_fdflags="" # it appears is no needed...
	try_f90flags="-x f95-cpp-input -J ../include/"
    try_openmp="-fopenmp"
    try_phigemm_opts="-D__GFORTRAN -D__STD_F95"
    have_cpp=0 #???
	;;
crayxt*:pgf* )
	try_ccflags="-O3 -mp -fPIC"
	try_ldflags="-O3 -mp -fPIC"
	try_ldsharedflag="-shared"
	pre_fdflags="-Mpreprocess"
	try_f90flags="-module ../include/"
    try_openmp="-mp"
    try_phigemm_opts="-D__PGI -D__CRAY"
    have_cpp=0 #???
	;;
* )
	# ABORT
	;;
esac

if test "$use_multithread" -eq 1 ; then
  try_ccflags="$try_ccflags $try_openmp"
  try_ldflags="$try_ldflags $try_openmp"
  try_f90flags="$try_f90flags $try_openmp"
fi

if test "$ccflags"  = "" ; then ccflags=$try_ccflags   ; fi
if test "$ldflags"  = "" ; then ldflags=$try_ldflags   ; fi
if test "$f90flags" = "" ; then f90flags=$try_f90flags ; fi

ldsharedflag=$try_ldsharedflag


# linker and archiver
# note that from this point on, further additions to
# linker flags should be added to ldflags rather than try_ldflags
if test "$ld" = ""       ; then ld="$cc"           ; fi
if test "$ar" = ""       ; then ar="$try_ar"           ; fi
if test "$arflags" = ""  ; then arflags="$try_arflags" ; fi

echo setting LD... $ld
echo setting LDFLAGS... $ldflags

echo setting AR... $ar
echo setting ARFLAGS... $arflags

ld_library_path=`echo $LD_LIBRARY_PATH | sed 's/:/ /g'`

AC_LANG_PUSH(C)

# Looking for NVCC & GCC
AC_PATH_PROG(NVCC, nvcc, )
AC_ARG_VAR([NVCC], [nvcc compiler to use])
have_valid_cuda_compiler=1
AC_PATH_PROG(NVCC, nvcc, )
if test -n "$nvcc"; then
    have_valid_cuda_compiler=0
fi

AC_PATH_PROG(GNUCC, gcc, )
if test -n "$gnucc"; then
    AC_MSG_ERROR([Please check if GNU compilers are correctly installed, gcc compiler missing in PATH.])
fi
AC_PATH_PROG(GNUFC, gfortran, )
if test -n "$gnufc"; then
    AC_MSG_ERROR([Please check if GNU compilers are correctly installed, gfortran compiler missing in PATH.])
fi

# Looking for GEMM libraries...  
have_gemm=0
try_libdirs="$blas_lib_path $libdirs $ld_library_path"
for dir in none $try_libdirs
do
    unset ac_cv_search_dgemm # clear cached value
    
    if test "$dir" = "none"
    then
        try_loption=" "
    else
        echo $ECHO_N "in $dir: " $ECHO_C
        try_loption="-L$dir"
    fi
    
    FFLAGS=""
    LIBS=""
    GEMM_LIB=""

	case "$arch:$f90_version" in
	ia32:ifort* | ia64:ifort* | x86_64:ifort*  )
		
		# Testing for ACML...
        if test "$use_multithread" -eq 0; then
            for dir in $try_libdirs
            do
                GEMM_LIB="$dir/libacml.a"
                AC_CHECK_FILE($GEMM_LIB, try_phigemm_opts="$try_phigemm_opts -D__ACML" have_gemm=1 blas_libs="$GEMM_LIB $dir/libifcoremt_pic.a -lrt",)
                
                if test "$have_gemm" -eq 1
                then break ; fi
            done
        else
            for dir in $try_libdirs
            do
                GEMM_LIB="$dir/libacml_mp.a"
                AC_CHECK_FILE($GEMM_LIB, try_phigemm_opts="$try_phigemm_opts -D__ACML -D__OMP" have_gemm=1 blas_libs="$GEMM_LIB $dir/libifcoremt_pic.a -lrt -liomp5 -lpthread",)
                
                if test "$have_gemm" -eq 1
                then break ; fi
            done
        fi
        
        GEMM_LIB=""
        
        # Testing for MKL...
        if test "$use_multithread" -eq 0; then
            AC_SEARCH_LIBS(dgemm, mkl_intel_lp64, try_phigemm_opts="$try_phigemm_opts -D__MKL" have_gemm=1
               blas_libs="$try_loption $LIBS -lmkl_sequential -lmkl_core -lpthread"
               ldflags="$GEMM_LIB $ldflags", echo "MKL not found",
               -lmkl_sequential -lmkl_core)
        else
            LDFLAGS="$try_loption -openmp"
            AC_SEARCH_LIBS(dgemm, mkl_intel_lp64, try_phigemm_opts="$try_phigemm_opts -D__MKL -D__OMP" have_gemm=1
               blas_libs="$try_loption $LIBS -lmkl_intel_thread -lmkl_core -liomp5 -lpthread"
               ldflags="$GEMM_LIB $ldflags", echo "MKL not found",
               -lmkl_intel_thread -lmkl_core)
        fi
        
        if test "$have_gemm" -eq 1
        then break ; fi
	    ;;
	ia32:pgf* | ia64:pgf* | x86_64:pgf* )
	
	    # Testing for ACML...
        if test "$use_multithread" -eq 0; then
            for dir in $try_libdirs
            do
                GEMM_LIB="$dir/libacml.a"
                AC_CHECK_FILE($GEMM_LIB, try_phigemm_opts="$try_phigemm_opts -D__ACML" have_gemm=1 blas_libs="$GEMM_LIB -lrt",)
                
                if test "$have_gemm" -eq 1
                then break ; fi
            done
        else
            for dir in $try_libdirs
            do
                GEMM_LIB="$dir/libacml_mp.a"
                AC_CHECK_FILE($GEMM_LIB, try_phigemm_opts="$try_phigemm_opts -D__ACML -D__OMP" have_gemm=1 blas_libs="$GEMM_LIB -lrt",)
                
                if test "$have_gemm" -eq 1
                then break ; fi
            done
        fi
        
        GEMM_LIB=""
        
        # Testing for MKL...
        if test "$use_multithread" -eq 0; then
            AC_SEARCH_LIBS(dgemm, mkl_intel_lp64, try_phigemm_opts="$try_phigemm_opts -D__MKL" have_gemm=1
               blas_libs="$try_loption $LIBS -lmkl_sequential -lmkl_core -lpthread"
               ldflags="$GEMM_LIB $ldflags", echo "MKL not found",
               -lmkl_sequential -lmkl_core -lpthread -lm)
        else
            LDFLAGS="$try_loption -mp"
            AC_SEARCH_LIBS(dgemm, mkl_intel_lp64, try_phigemm_opts="$try_phigemm_opts -D__MKL -D__OMP" have_gemm=1
               blas_libs="$try_loption $LIBS -lmkl_pgi_thread -lmkl_core -lpthread -lm -pgf90libs"
               ldflags="$GEMM_LIB $ldflags", echo "MKL not found",
               -lmkl_pgi_thread -lmkl_core -lpthread -lm -pgf90libs)
        fi
        
        if test "$have_gemm" -eq 1
        then break ; fi
		;;
	*:gfortran )
		# Testing for ACML...
        if test "$use_multithread" -eq 0; then
            for dir in $try_libdirs
            do
                GEMM_LIB="$dir/libacml.a"
                AC_CHECK_FILE($GEMM_LIB, try_phigemm_opts="$try_phigemm_opts -D__ACML" have_gemm=1 blas_libs="$GEMM_LIB -lgfortran -lm -lrt",)
                
                if test "$have_gemm" -eq 1
                then break ; fi
            done
        else
            for dir in $try_libdirs
            do
                GEMM_LIB="$dir/libacml_mp.a"
                AC_CHECK_FILE($GEMM_LIB, try_phigemm_opts="$try_phigemm_opts -D__ACML -D__OMP" have_gemm=1 blas_libs="$GEMM_LIB -lgfortran -lgomp -lm -lrt",)
                
                if test "$have_gemm" -eq 1
                then break ; fi
            done
        fi
        
        GEMM_LIB=""
        
        # Testing for MKL...
        if test "$use_multithread" -eq 0; then
            AC_SEARCH_LIBS(dgemm, mkl_intel_lp64, try_phigemm_opts="$try_phigemm_opts -D__MKL" have_gemm=1
               blas_libs="$try_loption $LIBS -lmkl_sequential -lmkl_core -lpthread -lm"
               ldflags="$GEMM_LIB $ldflags", echo "MKL not found",
               -lmkl_sequential -lmkl_core -lpthread -lm)
        else
            LDFLAGS="$try_loption -fopenmp"
            AC_SEARCH_LIBS(dgemm, mkl_intel_lp64, try_phigemm_opts="$try_phigemm_opts -D__MKL -D__OMP" have_gemm=1
               blas_libs="$try_loption $LIBS -lmkl_gnu_thread -lmkl_core -fopenmp -lpthread -lm"
               ldflags="$GEMM_LIB $ldflags", echo "MKL not found",
               -lmkl_gnu_thread -lmkl_core -fopenmp -lpthread -lm)
        fi
        if test "$have_gemm" -eq 1
        then break ; fi
		;;
	crayxt* )
	    # Technically the cc/ftn wrappers should take care of everything...
	    have_gemm=1
		
	    # Note: __CRAY -> __ACML_MP or __ACML or __LIBSCI
	    #       but there might MKL active! let's detect it...
	    if test "$use_multithread" -eq 0; then
            AC_SEARCH_LIBS(dgemm, mkl_intel_lp64, try_phigemm_opts="$try_phigemm_opts -D__MKL", echo "MKL not found")
        else
            LDFLAGS="$try_loption -openmp"
            AC_SEARCH_LIBS(dgemm, mkl_intel_lp64, try_phigemm_opts="$try_phigemm_opts -D__MKL -D__OMP", echo "MKL not found")
        fi
	    ;;
	* )
		# ABORT
		;;
	esac
                        
    if test "$have_gemm" -eq 1
    then break ; fi
done

if test "$have_gemm" -eq 0  ; then
    #AC_MSG_WARN([Please check if a valid BLAS library is correctly installed, *GEMM symbols were not automatically detected.])
    AC_MSG_ERROR([Please check if a valid BLAS library is correctly installed, *GEMM symbols were not automatically detected.])
fi

# Search for libraries
# Flags for needed libraries
have_cuda=0
have_magma=0

# Warning
#if test "$use_magmablas" -eq 1; then
#    AC_MSG_WARN([MAGMA support has not yet fully implemented, stay tuned for the next version.])
#    use_magmablas=0
#fi

# check for CUDA headers
have_valid_cuda_header=0
if test -e "$cuda_path/include/cuda.h"; then
    have_valid_cuda_header=1
fi

# check for CUDA libs
if test "$have_valid_cuda_header" == 1  && test "$have_valid_cuda_compiler" == 1 
then 
	nvcc=$NVCC
	
	try_libdirs="$cuda_path/lib64 /usr/lib64/nvidia"
    try_libdirs="$try_libdirs $libdirs $ld_library_path"
    for dir in $try_libdirs
    do
        unset ac_cv_search_cudaGetDeviceCount # clear cached value
        unset ac_cv_search_cublasInit
        unset ac_cv_search_cudaMemGetInfo
        
        if test "$dir" = "none"
        then
                try_loption=
        else                  
                try_loption="-L$dir"
        fi	
		
        if test "$have_cuda" -eq 0 ; then
			CFLAGS="$test_cflags"
			LDFLAGS="$try_loption"
	        LIBS=""    
	        echo $ECHO_N "in $dir: " $ECHO_C				
			AC_SEARCH_LIBS(cudaGetDeviceCount,cudart, 
			      cuda_libs="$cuda_libs $LIBS",,)
			LIBS=""
			
			echo $ECHO_N "in $dir: " $ECHO_C	
			AC_SEARCH_LIBS(cublasInit,cublas, 
				cuda_libs="$cuda_libs $LIBS",,)			  
	 	 			  
	 		cuda_libs="$try_loption $cuda_libs"
	 	
			if test "$ac_cv_search_cudaGetDeviceCount" != "no" && 
				test "$ac_cv_search_cublasInit" != "no"
			then			
				have_cuda=1
			fi
		fi
		
		if test "$have_cuda" -eq 1
		then break ; fi
		
	done   	
else
    AC_MSG_ERROR([Please check if CUDA is correctly installed.]) 
fi

# check for MAGMABLAS 
# ATTENTION: MAGMA requires CUBLAS ** ALWAYS **	 
if test "$have_cuda" -eq 1  && test "$use_magmablas" -eq 1 
then 

    try_libdirs="$magma_path $libdirs $ld_library_path"
	
    for dir in $try_libdirs
    do        
        MAGMA_LIB="$dir/libmagmablas.a"
        AC_CHECK_FILE($MAGMA_LIB, have_magma=1 cuda_libs="$MAGMA_LIB $cuda_libs",)
		
		if test "$have_magma" -eq 1
		then break ; fi
	done   	
fi

#?
AC_LANG_POP(C)  	

# Warning message ...
if test "$use_magmablas" -eq 1 && test "$have_magma" -eq 0 ; then
    AC_MSG_WARN([MAGMA has not been detected, CUBLAS will be used instead.])
fi
	
if test "$use_multithread" -eq 1; then
	nvccflags="-O3 -ccbin gcc --compiler-options '-fPIC -fopenmp'"
else
	nvccflags="-O3 -ccbin gcc --compiler-options '-fPIC'"
fi
		
# *THIS SECTION CAN BE GENERALIZED*
# GPU architecture (default is 20)
case "$cuda_arch" in
35 )
	# This is for TESLA K20 cards (GK110)
    nvccflags="$nvccflags -m64 -arch=sm_35 --ptxas-options=-v --fmad=false"
    #try_phigemm_opts="$try_phigemm_opts -D__GPU_NVIDIA_30"
    ;;
30 )
	# This is for GeForce GTX6xx and TESLA K10 cards (GK104)
    nvccflags="$nvccflags -m64 -arch=sm_30 --ptxas-options=-v --fmad=false"
    #try_phigemm_opts="$try_phigemm_opts -D__GPU_NVIDIA_30"
    ;;
25 )
    nvccflags="$nvccflags -m64 -arch=sm_25 --ptxas-options=-v --fmad=false"
    #try_phigemm_opts="$try_phigemm_opts -D__GPU_NVIDIA_20"
    ;;
20 )
    nvccflags="$nvccflags -m64 -arch=sm_20 --ptxas-options=-v --fmad=false"
    #try_phigemm_opts="$try_phigemm_opts -D__GPU_NVIDIA_20"
    ;;
13 )
    nvccflags="$nvccflags -m64 -arch=sm_13 --ptxas-options=-v"
    #try_phigemm_opts="$try_phigemm_opts -D__GPU_NVIDIA_13"
    ;;        
esac


if test "$use_debug" == 1 ; then
    try_phigemm_opts="$try_phigemm_opts -D__PHIGEMM_DEBUG"
fi

if test "$use_multigpu" == 1 ; then
    try_phigemm_opts="$try_phigemm_opts -D__PHIGEMM_MULTI_GPU -D__PHIGEMM_PINNED"
else
	if test "$use_pinned" == 1 ; then
    	try_phigemm_opts="$try_phigemm_opts -D__PHIGEMM_PINNED"
	fi
fi

if test "$use_profile" == 1 ; then
    try_phigemm_opts="$try_phigemm_opts -D__PHIGEMM_PROFILE"
fi

if test "$use_parallel" == 1 ; then
    try_phigemm_opts="$try_phigemm_opts -D__PHIGEMM_PARA"
fi

if test "$use_explicit_factor" == 1 ; then
    try_phigemm_opts="$try_phigemm_opts -D__PHIGEMM_EXPLICIT_SPLITFACTOR"
fi
	
if test "$use_selftune" -eq 1 ; then
    try_phigemm_opts="$try_phigemm_opts -D__PHIGEMM_SELFTUNE"
fi	

if test "$use_magmablas" -eq 1 && test "$have_magma" -eq 1 ; then
    try_phigemm_opts="$try_phigemm_opts -D__PHIGEMM_MAGMABLAS"
else
    try_phigemm_opts="$try_phigemm_opts -D__PHIGEMM_CUBLAS"	
fi

if test "$use_cpu_only" -eq 1 ; then
    try_phigemm_opts="$try_phigemm_opts -D__PHIGEMM_CPUONLY"
fi

if test "$use_gpu_only" -eq 1; then
    try_phigemm_opts="$try_phigemm_opts -D__PHIGEMM_GPUONLY"
fi

if test "$use_special_k" -eq 1; then
    try_phigemm_opts="$try_phigemm_opts -D__PHIGEMM_ENABLE_SPECIALK"
fi

phigemm_opts=$try_phigemm_opts

cuda_include="-I$cuda_path/include -I$topdir/include"
# blas_libs="$topdir/lib/libphigemm.a $blas_libs"   

echo setting CUDA_LIBS... $cuda_libs

# ?	
try_iflags="$try_iflags $cuda_include"

ld_libs="$topdir/lib/libphigemm.a $cuda_libs $blas_libs"

if test "$have_cuda" -eq 1; then
   cuda_line="CUDA_LIBS=$cuda_libs"
else
   cuda_line=""
   cuda_line="@delete@"
fi

# export settings to generated files
AC_SUBST(cc)
AC_SUBST(ccflags)
AC_SUBST(cpp)
AC_SUBST(cppflags)
AC_SUBST(f90)
AC_SUBST(f90flags)
AC_SUBST(ld)
AC_SUBST(ldflags)
AC_SUBST(ldsharedflag)
AC_SUBST(pre_fdflags)
AC_SUBST(ld_libs)
AC_SUBST(ar)
AC_SUBST(arflags)
AC_SUBST(nvcc)
AC_SUBST(nvccflags)
AC_SUBST(nvccdflags)
AC_SUBST(cuda_include)
AC_SUBST(cuda_libs)
AC_SUBST(phigemm_opts)
AC_SUBST(topdir)
AC_SUBST(cuda_path)

AC_CONFIG_FILES(make.inc)
AC_OUTPUT

