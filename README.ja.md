LAPACK/BLAS 入門：サンプルプログラム
============================================================

2016-12-02 (Fri) 幸谷 智紀
---------------------------------

　本文及び演習問題で使われているプログラムはLinux環境下で実行を確認したものです。Windows環境下ではCygwinを使うことで同等の環境を整えることができます。Windows環境下での，Visual C++&Intel C++ compilerを使ったBLAS, LAPACK用プログラムのコンパイルも可能です。

☆サンプルプログラム一覧
-----------------------------

### 共通  
>	lapack_gcc.inc ... GCC用設定ファイル (Linux)  
>	lapack_icc.inc ... Intel C compiler用設定ファイル (Linux)  
>	lapack_win_intel.inc ... Intel C compiler用設定ファイル (Windows)  
>	Makefile.unix ... Linux用メイクファイル → GCCかIntel C compilerかを選び，設定ファイルを読み込ませて，"make -f Makefile.unix"で生成  
>	Makefile.win_intel ... Windows用メイクファイル → Intel C compilerの設定ファイルを読み込ませて，"make -f Makefile.win_intel"で生成  
>	windows\ ... Windows環境下でCBLAS, LAPACKEのインクルードファイルを置いておくフォルダ  

### 第1章
>	first.c ... 単精度，倍精度基本演算と相対誤差の導出  
>	complex_first.c ... 複素数演算(C言語用)  
>	complex_first_cpp.c ... 複素数演算(C++用)  

### 第2章
>	my_matvec_mul.c ... 行列・ベクトル積  
>	matvec_mul.c ... DGEMVを用いた実行列・ベクトル積  
>	complex_matvec_mul.c ... ZGEMVを用いた複素行列・ベクトル積  
>	my_linear_eq.c ... 連立一次方程式の求解  
>	linear_eq.c ... DGESVを用いた連立一次方程式の求解  
>	row_column_major.c ... 行優先，列優先行列格納形式  
>	complex_row_column_major.c ... 複素数行優先，列優先行列格納形式  
>	lapack_complex_row_column_major.c ... LAPACK関数を用いた複素数行優先，列優先行列格納形式  
>	lapack_complex_row_column_major.cc ... LAPACK関数を用いた複素数行優先，列優先行列格納形式(C++)  

### 第3章
>	blas1.c ... BLAS1関数サンプル  
>	blas2.c ... BLAS2関数サンプル  
>	blas3.c ... BLAS3関数サンプル  
>	jacobi_iteration.c ... ヤコビ反復法  
>	power_eig.c ... べき乗法  

### 第4章
>	linear_eq_dgetrf.c ... LU分解，前進代入・後退代入  
>	linear_eq_dsgesv.c ... 混合精度反復改良法  
>	linear_eq_dsposv.c ... 実対称行列用の混合精度反復改良法  
>	lapack_dgecon.c ... 条件数の計算  
>	lapack_lamch.c ... マシンイプシロン等の導出  
>	invpower_eig.c ... 逆べき乗法  
>	lapack_dgeev.c ... 実非対称行列用固有値・固有ベクトル計算  
>	lapack_dsyev.c ... 実対称行列用固有値・固有ベクトル計算  
>	lapack_ssyev.c ... 実対称行列用固有値・固有ベクトル計算(単精度)  

### 第5章
>	my_matvec_mul_pt.c ... Pthreadで並列化した行列・ベクトル積計算  
>	my_matvec_mul_omp.c ... OpenMPで並列化した行列・ベクトル積計算  
>	my_linear_eq_omp.c ... OpenMPで並列化したLU分解，前進代入・後退代入計算  

### 第6章
>	jacobi_iteration_mkl.c ... COO形式疎行列用のJacobi反復法  
>	jacobi_iteration_csr_mkl.c... CSR形式疎行列用のJacobi反復法  
>	bicgstab_mkl.c ... COO形式疎行列用のBiCGSTAB法  
>	bicgstab_csr_mkl.c ... CSR形式疎行列用のBiCGSTAB法  
>	mm/matrix_market_io.h  ... MatrixMarketフォーマット用関数定義  
>	mm/matrix_market_io.c  ... MatrixMarketフォーマット用関数群  

### 第7章 (Windows環境下の実行はサポートしていません)
>	mycuda_daxpy.cu ... CUDAサンプルプログラム  
>	matvec_mul_cublas.c ... CUBLASを用いた行列・ベクトル積  
>	matvec_mul_magma.c ... MAGMAとCUBLASを用いた行列・ベクトル積  
>	matvec_mul_magma_pure.c ... MAGMAだけを用いた行列・ベクトル積  
>	linear_eq_magma.c ... MAGMAを用いた連立一次方程式の求解  
>	lapack_dgeev_magma.c ... MAGMAを用いた実非対称行列用固有値・固有ベクトル計算  
>	bicgstab_csr_cusparse.c ... cuSPARSEを用いたBiCGSTAB法  

### 第8章
>	integral_eq/Makefile.unix ... 積分方程式求解プログラムのコンパイル(Linux)  
>	integral_eq/Makefile.win_intel ... 積分方程式求解プログラムのコンパイル(Windows)  
>	integral_eq/gauss_integral.h ... ガウス積分公式導出のためのヘッダファイル  
>	integral_eq/gauss_integral.c ... ガウス積分公式の導出  
>	integral_eq/iteration.c ... 割線法とデリバティブフリー解法  

☆コンパイル条件
-----------------------------

　本プログラムはLinux, Windowsソフトウェア開発環境下でコンパイル＆実行可能であることを下記の環境で確認しております。

・Linux   ... GCC 4.4.7, Intel C/C++/Fortran compiler 13.1.3, Intel Math Kernel Library 11.0.5, LAPACK 3.6.0, MAGMA 1.6.0, CUDA 7.5 on CentOS 6.5 x86_64  
          ... GCC 4.4.7, Iitel C/C++/Fortran compiler 14.0.2, Intel Math Kernel Library 11.1.2, LAPACK 3.6.0, MAGMA 1.6.1, CUDA 7.0 on CentOS 6.3 x86_64  
・Windows ... Intel C++ compiler 16.0.1.146, Intel Math Kernel Library 11.3.2 on Windows 8.1 x64  

　Linux, Windows環境下での本プログラムのコンパイル＆実行は，上記のソフトウェア環境が整っているCUIで行って下さい。それ以外の環境下での諸問題については確認ができませんので，お答えすることも不可能です。


☆コンパイル方法 ... Linux
-----------------------------

0. Intel Math Kernel, CUDA, MAGMA, LAPACKE/CBLASをインストールし，インストール先のディレクトリをMakefile.unix内の適切なマクロ名に設定  
1. Intel C/C++ Compilerの場合はlapack_icc.incを，GCCの場合はlapack_gcc.incを環境に合わせて修正し，それぞれのコンパイラが適切に動作するよう環境設定を行い，Makefile.unixが読み込むファイルを設定  
2. make -f ./Makefile.unix でコンパイル  
3. make -f ./Makefile.unix clean でobjectファイル，実行ファイルが消去される  

☆コンパイル方法 ... Windows
-----------------------------

0. Intel Math Kernelをインストールし，インストール先のディレクトリをMakefile.win_intel内の適切なマクロ名に設定。また，LAPACKEとCBLASのインクルードファイルをwindowsフォルダにコピーしておく（デフォルト設定の場合）。  
1. Intel C/C++ CompilerとVisual C++が適切に動作するよう環境設定を行う  
2. nmake -f ./Makefile.win_intel でコンパイル  
3. nmake -f ./Makefile.win_intel clean でobjectファイル，実行ファイルが消去される  

