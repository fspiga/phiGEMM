!*****************************************************************************\
!* Copyright (C) 2011-2014 Quantum ESPRESSO Foundation
!* Copyright (C) 2010-2011 Irish Centre for High-End Computing (ICHEC)
!*
!* This file is distributed under the terms of the
!* GNU General Public License. See the file `License'
!* in the root directory of the present distribution,
!* or http://www.gnu.org/copyleft/gpl.txt .
!*
!* Filippo Spiga (filippo.spiga@quantum-espresso.org)
!*****************************************************************************/

module phigemm

  implicit none

#if defined(__PHIGEMM_WEAK_INTERFACES)
    EXTERNAL phisgemm
    EXTERNAL phidgemm
    EXTERNAL phigemmsetsplitfactor
#else

  interface

    subroutine phigemmsetsplitfactor (split_dgemm, split_zgemm)
       real               :: split_dgemm
       real               :: split_zgemm
    end subroutine phigemmsetsplitfactor

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
