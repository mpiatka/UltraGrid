# see https://en.opensuse.org/openSUSE:Build_Service_Debian_builds#packageName.dsc
DEBTRANSFORM-TAR:	ndi4.tar.gz
DEBTRANSFORM-FILES-TAR:	debian.tar.gz
DEBTRANSFORM-SERIES:	debian-patches-series
Format: 1.0
Source: ultragrid-proprietary-drivers-ndi
Binary: ultragrid-proprietary-drivers-ndi
Architecture: any
Version: 20200227
Standards-Version: 3.9.6
Maintainer: 	Lukas Rucka <ultragrid-dev@cesnet.cz>
Build-Depends: 	debhelper (>= 7.0.50~), build-essential, linux-headers, realpath, coreutils, autoconf, automake, linux-libc-dev, bash
