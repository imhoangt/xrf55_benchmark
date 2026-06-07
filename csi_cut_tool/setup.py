from setuptools import setup

setup(
    name='csi_tool',
    version='1.0.0',
    description='Thư viện đồng bộ và trích xuất dữ liệu CSI (1 file lõi)',
    py_modules=['csi_tool'],                 # Khai báo trực tiếp tên file (không có đuôi .py)
    install_requires=[                       
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'openpyxl'                           
    ],
    python_requires='>=3.8',
)