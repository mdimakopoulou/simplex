#ifndef _SIMPLEX_MAIN_H
#define _SIMPLEX_MAIN_H

#include <cstdint>


#if __SIZEOF_INT__ != 4
#   error "Cannot run on this architecture (int is not 32bits)"
#endif

#if __SIZEOF_DOUBLE__ != 8 
#   error "Cannot run on this architecture (double is not 64bits)"
#endif

extern int verbose;

std::string loadTextFile(const char* filename);

#endif
