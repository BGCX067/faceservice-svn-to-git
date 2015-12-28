AC_INIT([faceservice], [0.2], [arjan@scherpenisse.net])
AM_CONFIG_HEADER(config.h)
AM_INIT_AUTOMAKE
AC_CONFIG_MACRO_DIR([.])
m4_ifdef([LT_INIT],[LT_INIT],[AC_PROG_LIBTOOL])
AC_LANG_CPLUSPLUS
AC_PROG_CC
AC_PROG_CXX

PKG_CHECK_MODULES(OPENCV, opencv,,exit)
dnl AC_CHECK_LIB(highgui,cvLoadImage,,exit)
AC_SUBST(OPENCV_LIBS)

report_curl="no"
PKG_CHECK_MODULES(CURL, libcurl,,exit )
#CURL_LIBS="-lcurl"
AC_SUBST(CURL_LIBS)

dnl Info on installed cgicc
AC_ARG_WITH(cgicc,
  [  --with-cgicc[=DIR]      use cgicc installation in DIR],
  [if ! test -d "${withval}"; then
    AC_MSG_ERROR(bad value '${withval}' for --with-cgicc: not a directory)
   fi
   cgicc_prefix=${withval}])



dnl Perform some sanity checks on the cgicc installation
CGICC_CHECK_INSTALLATION($cgicc_prefix)


AC_OUTPUT(Makefile src/Makefile)

AC_MSG_RESULT([
General configuration ================================================
Compiler: ${CXX}
CXXFLAGS: ${CXXFLAGS}

Install path: ${prefix}

======================================================================

Now run make ...
])
