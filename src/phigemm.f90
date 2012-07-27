! Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
! Copyright (C) 2011-2012 Quantum ESPRESSO Foundation
!
! This file is distributed under the terms of the
! GNU General Public License. See the file `License'
! in the root directory of the present distribution,
! or http://www.gnu.org/copyleft/gpl.txt .

module phigemm

  implicit none

#if defined(__PHIGEMM_WEAK_INTERFACES)
    EXTERNAL phisgemm
    EXTERNAL phidgemm
    EXTERNAL phicgemm
    EXTERNAL phizgemm
    EXTERNAL phigemmsetsplitfactor
#else

  !---- Fortran interfaces to phiGEMM subroutines ----
  interface

    subroutine phigemmsetsplitfactor (x)
       real               :: x(*)
    end subroutine phigemmsetsplitfactor

#if defined(__PHIGEMM_PROFILE)
     subroutine phiSgemm( transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, file, line)
       character          :: transa
       character          :: transb
       integer            :: m
       integer            :: n
       integer            :: k
       real               :: alpha
       real               :: A(*)
       integer            :: lda
       real               :: B(*)
       integer            :: ldb
       real               :: beta
       real               :: C(*)
       integer            :: ldc
       character(len = *) :: file
       character(len = *) :: line
     end subroutine phiSgemm
#else
     subroutine phiSgemm( transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
       character       :: transa
       character       :: transb
       integer         :: m
       integer         :: n
       integer         :: k
       real            :: alpha
       real            :: A(*)
       integer         :: lda
       real            :: B(*)
       integer         :: ldb
       real            :: beta
       real            :: C(*)
       integer         :: ldc
     end subroutine phiSgemm
#endif


#if defined(__PHIGEMM_PROFILE)
     subroutine phiCgemm( transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, file, line)
       character       :: transa
       character       :: transb
       integer         :: m
       integer         :: n
       integer         :: k
       complex         :: alpha
       complex         :: A(*)
       integer         :: lda
       complex         :: B(*)
       integer         :: ldb
       complex         :: beta
       complex         :: C(*)
       integer         :: ldc
       character(len = *) :: file
       character(len = *) :: line
     end subroutine phiCgemm
#else
     subroutine phiCgemm( transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
       character       :: transa
       character       :: transb
       integer         :: m
       integer         :: n
       integer         :: k
       complex         :: alpha
       complex         :: A(*)
       integer         :: lda
       complex         :: B(*)
       integer         :: ldb
       complex         :: beta
       complex         :: C(*)
       integer         :: ldc
     end subroutine phiCgemm
#endif


#if defined(__PHIGEMM_PROFILE)
     subroutine phiDgemm( transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, file, line)
       character       :: transa
       character       :: transb
       integer         :: m
       integer         :: n
       integer         :: k
       double precision:: alpha
       double precision:: A(*)
       integer         :: lda
       double precision:: B(*)
       integer         :: ldb
       double precision:: beta
       double precision:: C(*)
       integer         :: ldc
       character(len = *) :: file
       character(len = *) :: line
     end subroutine phiDgemm
#else
     subroutine phiDgemm( transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
       character       :: transa
       character       :: transb
       integer         :: m
       integer         :: n
       integer         :: k
       double precision:: alpha
       double precision:: A(*)
       integer         :: lda
       double precision:: B(*)
       integer         :: ldb
       double precision:: beta
       double precision:: C(*)
       integer         :: ldc
     end subroutine phiDgemm
#endif


#if defined(__PHIGEMM_PROFILE)
     subroutine phiZgemm( transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, file, line)
       character       :: transa
       character       :: transb
       integer         :: m
       integer         :: n
       integer         :: k
       complex*16      :: alpha
       complex*16      :: A(*)
       integer         :: lda
       complex*16      :: B(*)
       integer         :: ldb
       complex*16      :: beta
       complex*16      :: C(*)
       integer         :: ldc
       character(len = *) :: file
       character(len = *) :: line
     end subroutine phiZgemm
#else
     subroutine phiZgemm( transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
       character       :: transa
       character       :: transb
       integer         :: m
       integer         :: n
       integer         :: k
       complex*16      :: alpha
       complex*16      :: A(*)
       integer         :: lda
       complex*16      :: B(*)
       integer         :: ldb
       complex*16      :: beta
       complex*16      :: C(*)
       integer         :: ldc
     end subroutine phiZgemm
#endif

  end interface

#endif

end module phigemm
