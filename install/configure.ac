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
   
AC_ARG_ENABLE(pinned, 
   [AS_HELP_STRING([--enable-pinned],
       [enable multi GPU support (default: no)])],
   [if  test "$enableval" = "yes" ; then
      use_pinned=1
   else
      use_pinned=0
   fi],
   [use_pinned=0])

AC_ARG_ENABLE(self_tune, 
   [AS_HELP_STRING([--enable-self-tune],
       [enable self-tune (default: no, experimental)])],
   [if  test "$enableval" = "yes" ; then
      use_selftune=1
   else
      use_selftune=0
   fi],
   [use_selftune=0])
      
AC_ARG_WITH(cuda_dir, 
   [AS_HELP_STRING([--with-cuda-dir=<path>],
    [specify CUDA installation directory (default is /usr/local/cuda/)])],
    [cuda_path="$withval"], 
    [cuda_path="/usr/local/cuda/"])
        
AC_ARG_WITH(split_by_hand,
   [AS_HELP_STRING([--with-split-by-hand],
       [the split factor has to be specified in the code instead of using env variables (default: no)])],
    [if   test "$withval" = "yes" ; then
      use_explicit_factor=1
   else
      use_explicit_factor=0
   fi],
   [use_explicit_factor=0])

AC_ARG_WITH(cpu_only,
   [AS_HELP_STRING([--with-cpu_only],
       [everything is performed only by the CPU, useful for call-by-call profiling (default: no)])],
    [if   test "$withval" = "yes" ; then
      use_cpu_only=1
   else
      use_cpu_only=0
   fi],
   [use_cpu_only=0])
   
# -----------------------------
# store variables from the environment, if set (may be or not be set)
# If set, they take precedence over configure internal choice.
# Flags and libraries are accepted without further testing;
# compilers are tested. Specify compiler name only, not the full path
# (i.e. F90=/usr/local/bin/f90 may not work, use F90=f90)
topdir=$TOPDIR     # current directory
arch=$ARCH         # see below for recognized architectures
env_cc=$CC         # C compiler (must be in the execution path)
cflags=$CFLAGS     # Flags for C compiler
cppflags=$CPPFLAGS # Flags for C preprocessor
ld=$LD             # Loader (must be in the execution path)
ldflags=$LDFLAGS   # Flags for loader
ld_libs=$LD_LIBS   # Additional libraries
blas_libs=$BLAS_LIBS     # blas library - specify e.g. /my/blas/lib/libmyblas.a
                         # or -L/my/blas/lib -lmyblas
libdirs=$LIBDIRS         # Where to look for libraries (e.g. /my/blas/lib)
ar=$AR                   # ar (shouldn't be needed)
arflags=$ARFLAGS         # Flags for ar (as above)
nvcc=$NVCC	 		# NVIDIA CUDA compiler 
nvccflags=$NVCCFLAGS	 		# Flags for NVIDIA CUDA compiler

# configure for current directory by default
if test "$topdir" = "" ; then topdir="`pwd`" ; fi

# check system type (no cross-compilation for now)
AC_CANONICAL_BUILD

# many HPC systems are configured so that running parallel programs
# interactively is disabled: on those systems, AC_PROG_FC / _F77 / _CC
# would fail because they can't run the compiled executables.
# to work around that, let's pretend we are cross-compiling even if we aren't
# !!! this relies on undocumented Autoconf behavior !!!

# This is used to distinguish between true and fake cross compilation
# (only on NEC SX8 actually)
if test "$host" != "" ; then ranlib=echo; fi

cross_compiling=yes
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

# Search for libraries
# Flags for needed libraries
have_cuda=0

# candidate archiver and archiver flags
try_ar="ar"
try_arflags="ruv"

# check serial Fortran 90 compiler. This must be done before performing
# the check for the parallel compiler (section below) because option
# --disable-parallel will do very strange things otherwise. The reason
# seems to be that autoconf does not repeat all tests for the second
# occurrence of AC_PROG_FC. So the first occurrence is the one that
# must always be performed, the second is optional. PG & CC sep.2006


try_f90="ifort ftn gfortran"
try_cc="icc ecc cc icc"

# use F90 if set
if test "$f90" = "" ; then f90="$try_f90" ; fi
AC_PROG_FC($f90)
f90=$FC

echo setting F90... $f90

# check serial C compiler
if test "$env_cc" = "" ; then cc="$try_cc" ; else cc="$env_cc"; fi
AC_PROG_CC($cc)
cc=$CC

echo setting CC... $cc


# linker and archiver
# note that from this point on, further additions to
# linker flags should be added to ldflags rather than try_ldflags
if test "$ld" = ""       ; then ld="$cc"           ; fi
if test "$ar" = ""       ; then ar="$try_ar"           ; fi
if test "$arflags" = ""  ; then arflags="$try_arflags" ; fi
#echo setting LD... $ld
#echo setting LDFLAGS... $ldflags
echo setting AR... $ar
echo setting ARFLAGS... $arflags

ld_library_path=`echo $LD_LIBRARY_PATH | sed 's/:/ /g'`

AC_LANG_PUSH(C)

# check for MKL blas
try_libdirs="/opt/intel/Compiler/*/*/mkl/lib/em64t
                 /opt/intel/mkl/*/lib/em64t
                 /opt/intel/mkl*/lib/em64t"
    try_libdirs="$libdirs $try_libdirs $ld_library_path"

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
            MKL_LIBS=""
            FFLAGS=""
            LDFLAGS="$try_loption -openmp"
            LIBS="$MKL_LIBS"
            #
            # should work for recent MKL versions only
            #
            if test "$use_multithread" -eq 0; then
   AC_SEARCH_LIBS(dgemm, mkl_intel_lp64,
                           have_mkl=1
                           blas_libs="$try_loption $LIBS -lmkl_sequential -lmkl_core -lpthread"
                           ldflags="$MKL_FLAGS $ldflags",
                           echo "MKL not found",
                           -lmkl_sequential -lmkl_core)
            else
   AC_SEARCH_LIBS(dgemm, mkl_intel_lp64,
                           have_mkl=1
                           blas_libs="$try_loption $LIBS -lmkl_intel_thread -lmkl_core -liomp5 -lpthread"
                           ldflags="$MKL_FLAGS $ldflags",
                           echo "MKL not found",
                           -lmkl_intel_thread -lmkl_core)
            fi
            if test "$ac_cv_search_dgemm" != "no"
            then break ; fi
    done

if test "$have_mkl" -eq 0  ; then
AC_MSG_ERROR([Please check if MKL libraries are correctly installed, DGEMM symbol missing.])
fi

# check for CUDA bin
AC_ARG_VAR([NVCC], [nvcc compiler to use])
have_valid_cuda_compiler=1
AC_PATH_PROG(NVCC, nvcc, )
if test -n "$nvcc"; then
    have_valid_cuda_compiler=0
fi

# check for CUDA headers
have_valid_cuda_header=0
if test -e "$cuda_path/include/cuda.h"; then
    have_valid_cuda_header=1
fi

if test "$have_valid_cuda_header" == 1  && 
    test "$have_valid_cuda_compiler" == 1 
then 
    # check for CUDA libs
	nvcc=$NVCC
	
	# AC_EGREP_HEADER does not work... yet
	# AC_EGREP_HEADER("4000", fileh, , try_dflags="$try_dflags -D__CUDA_3")
	val="`egrep "CUDA_VERSION 40" $cuda_path/include/cuda.h`"
	if test -z "$val" ; then
	   cudamajor=3
	   try_dflags="$try_dflags -D__CUDA_3"
	else
	   cudamajor=4
	fi
	
	try_libdirs="$cuda_path/lib64 /usr/lib64/nvidia"
    try_libdirs="$try_libdirs $libdirs $ld_library_path"
    for dir in $try_libdirs
    do
        unset ac_cv_search_cudaGetDeviceCount # clear cached value
        unset ac_cv_search_cublasInit
        unset ac_cv_search_cuMemGetInfo_v2
        if test "$dir" = "none"
        then
                try_loption=
        else                  
                try_loption="-L$dir"
        fi			
		CFLAGS="$test_cflags"
		# $test_ldflags
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
	 	
	 	LDFLAGS=""
	 	LIBS=""
        AC_SEARCH_LIBS(cuMemGetInfo_v2,cuda, 
              cuda_libs="$cuda_libs $LIBS",,)
              
		if test "$ac_cv_search_cudaGetDeviceCount" != "no" && 
			test "$ac_cv_search_cublasInit" != "no"
		then			
			have_cuda=1
			break
		fi
	done   	
	AC_LANG_POP(C)  	
else
    AC_MSG_ERROR([Please check if CUDA is correctly installed.]) 
fi


# GPU architecture
if  test "$cuda_arch" = "20" ; then
    nvccflags="-O3 --ptxas-options=-v -arch sm_20 -gencode arch=compute_20,code=compute_20"
fi
if test "$cuda_arch" = "13" ; then
    nvccflags="-O3 --ptxas-options=-v -arch sm_13 -gencode arch=compute_13,code=compute_13"
fi   

# Starting from empty flags....
phigemm_opts=

if test "$use_debug" == 1 ; then
    phigemm_opts="$phigemm_opts -D__PHIGEMM_DEBUG"
fi

if test "$use_multigpu" == 1 ; then
    phigemm_opts="$phigemm_opts -D__PHIGEMM_MULTI_GPU -D__PHIGEMM_PINNED"
else
	if test "$use_pinned" == 1 ; then
    	phigemm_opts="$phigemm_opts -D__PHIGEMM_PINNED"
	fi
fi

if test "$use_profile" == 1 ; then
    phigemm_opts="$phigemm_opts -D__PHIGEMM_PROFILE"
fi

if test "$use_parallel" == 1 ; then
    phigemm_opts="$phigemm_opts -D__PHIGEMM_PARA"
fi

if test "$use_explicit_factor" == 1 ; then
    phigemm_opts="$phigemm_opts -D__PHIGEMM_EXPLICIT_SPLITFACTOR"
fi
	
if test "$use_selftune" == 1 ; then
    phigemm_opts="$phigemm_opts -D__PHIGEMM_SELFTUNE"
fi	

if test "$use_cpu_only" == 1 ; then
    phigemm_opts="$phigemm_opts -D__PHIGEMM_HACK_CPUONLY"
fi
	       
cuda_include="-I$cuda_path/include -I$topdir/include"
blas_libs="$topdir/lib/libphigemm.a $blas_libs"   

try_iflags="$try_iflags $cuda_include"
ld_libs="$ld_libs $cuda_libs"

echo setting CUDA_LIBS... $cuda_libs
	
if test "$have_cuda" -eq 1; then
   cuda_line="CUDA_LIBS=$cuda_libs"
else
   cuda_line=""
   cuda_line="@delete@"
fi

# export settings to generated files
AC_SUBST(cc)
AC_SUBST(f90)
AC_SUBST(ld)
AC_SUBST(ld_libs)
AC_SUBST(blas_libs)
AC_SUBST(ar)
AC_SUBST(arflags)
AC_SUBST(nvcc)
AC_SUBST(nvccflags)
AC_SUBST(cudamajor)
AC_SUBST(cuda_include)
AC_SUBST(cuda_libs)
AC_SUBST(phigemm_opts)
AC_SUBST(topdir)
AC_SUBST(cuda_path)

AC_CONFIG_FILES(make.inc)

AC_OUTPUT