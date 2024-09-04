以下为虚拟环境中安装的所有库
库名   版本
brotli	1.1.0	
brotli-bin	1.1.0	
bzip2	1.0.8	
ca-certificates	2024.6.2	
certifi	2024.6.2	
contourpy	1.1.1	
cycler	0.12.1	
fonttools	4.53.0	
freetype	2.12.1	
glib	2.80.2	
glib-tools	2.80.2	
gst-plugins-base	1.24.4	
gstreamer	1.24.4	
icu	73.2	
importlib-resources	6.4.0	
importlib_resources	6.4.0	
intel-openmp	2024.1.0	
kiwisolver	1.4.5	
krb5	1.21.2	
lcms2	2.16	
lerc	4.0.0	
libblas	3.9.0	
libbrotlicommon	1.1.0	
libbrotlidec	1.1.0	
libbrotlienc	1.1.0	
libcblas	3.9.0	
libclang13	18.1.7	
libdeflate	1.20	
libffi	3.4.2	
libglib	2.80.2	
libhwloc	2.10.0	
libiconv	1.17	
libintl	0.22.5	
libintl-devel	0.22.5	
libjpeg-turbo	3.0.0	
liblapack	3.9.0	
libogg	1.3.4	
libpng	1.6.43	
libsqlite	3.46.0	
libtiff	4.6.0	
libvorbis	1.3.7	
libwebp-base	1.4.0	
libxcb	1.15	
libxml2	2.12.7	
libzlib	1.3.1	
m2w64-gcc-libgfortran	5.3.0	
m2w64-gcc-libs	5.3.0	
m2w64-gcc-libs-core	5.3.0	
m2w64-gmp	6.1.0	
m2w64-libwinpthread-git	5.0.0.4634.697f757	
matplotlib	3.7.3	
matplotlib-base	3.7.3	
mkl	2024.1.0	
msys2-conda-epoch	20160418	
munkres	1.1.4	
openjpeg	2.5.2	
openssl	3.3.1	
packaging	24.1	
pcre2	10.43	
pillow	10.3.0	
pip	24.0	
ply	3.11	
pthread-stubs	0.4	
pthreads-win32	2.9.1	
pyparsing	3.1.2	
pyqt	5.15.9	
pyqt5-sip	12.12.2	
python	3.8.19	
python-dateutil	2.9.0	
python_abi	3.8	
qt-main	5.15.8	
setuptools	70.0.0	
sip	6.7.12	
six	1.16.0	
tbb	2021.12.0	
tk	8.6.13	
toml	0.10.2	
tomli	2.0.1	
tornado	6.4.1	
ucrt	10.0.22621.0	
unicodedata2	15.1.0	
vc	14.3	
vc14_runtime	14.40.33810	
vs2015_runtime	14.40.33810	
wheel	0.43.0	
xorg-libxau	1.0.11	
xorg-libxdmcp	1.1.3	
xz	5.2.6	
zipp	3.19.2	
zstd	1.5.6

主要库为：numpy
pandas
sklearn
keras
matplotlib
tensorflow

运行步骤：
数据预处理并进行特征工程后修改文件中对应位置路径即可运行
run_by_folder.py为按照整个文件夹即病人类型运行，run_by_file.py则为单独预测一位病人
