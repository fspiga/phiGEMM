/*
  Copyright (C) 2002-2006 Quantum ESPRESSO group
  This file is distributed under the terms of the
  GNU General Public License. See the file `License'
  in the root directory of the present distribution,
  or http://www.gnu.org/copyleft/gpl.txt .
*/

#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

//#include "c_defs.h"

double cclock_ () {

    struct timeval tmp;
    double sec;
    gettimeofday( &tmp, (struct timezone *)0 );
    sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
    return sec;

}

double scnds_ ( ){
        static struct rusage T;

        getrusage(RUSAGE_SELF, &T);

        return ((double)T.ru_utime.tv_sec + ((double)T.ru_utime.tv_usec)/1000000.0);
}

