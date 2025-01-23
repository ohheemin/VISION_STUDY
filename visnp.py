import numpy as np

file_path = "/home/ohheemin/macaron_06/lidar_all_scans_noise.npy"


"""데이터를 로드하고, 파일로드를 성공하였는지, 데이터타입은 어떻게 되는지,
   배열 크기는 어떻게 되는지와 데이터 일부를 프린트한다. 로드하지 못했을 때는
   파일 로드를 실패하였다고 출력하게 한다. 
"""
try:
    data = np.load(file_path, allow_pickle=True)
    print("파일 로드 성공요")
    print("데이터 타입:", type(data))
    print("배열 크기:", data.shape if isinstance(data, np.ndarray) else "Not an ndarray")
    
    """배열의 데이터 값을 일부 값만 출력한다. 처음 항목의 5개만.
    """
    print("데이터 일부:", data[:5]) 
    
except Exception as e:
    print("파일 로드 실패요")
    print("오류 메시지:", e)

if data.size == 0:
    print("데이터가 비어 있음요")
