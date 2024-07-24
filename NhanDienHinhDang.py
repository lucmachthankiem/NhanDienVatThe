import cv2
import numpy as np

def nothing(x):
    pass

# Tạo một cửa sổ
cv2.namedWindow('Parameters')

# Tạo các thanh trượt để điều chỉnh giá trị ngưỡng và diện tích tối thiểu
cv2.createTrackbar('Threshold1', 'Parameters', 100, 255, nothing)
cv2.createTrackbar('Threshold2', 'Parameters', 200, 255, nothing)
cv2.createTrackbar('Min Area', 'Parameters', 5000, 30000, nothing)

def detect_shape(contour):
    # Tính chu vi của đường viền
    perimeter = cv2.arcLength(contour, True)
    
    # Tính gần đúng đa giác của đường viền
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    
    # Xác định hình dạng dựa trên số lượng đỉnh của đa giác
    if len(approx) == 3:
        return "Tam giac"
    elif len(approx) == 4:
        # Kiểm tra xem đó có phải là hình vuông hay hình chữ nhật
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        return "Hinh vuong" if 0.95 <= aspect_ratio <= 1.05 else "Hinh chu nhat"
    elif len(approx) == 5:
        return "Ngu giac"
    elif len(approx) > 5:
        return "Hinh tron"
    else:
        return "Vo dinh hinh"

# Mở camera
cap = cv2.VideoCapture(0)

while True:
    # Đọc một khung hình từ camera
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Chuyển đổi khung hình sang màu xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Lấy giá trị ngưỡng từ các thanh trượt
    thresh1 = cv2.getTrackbarPos('Threshold1', 'Parameters')
    thresh2 = cv2.getTrackbarPos('Threshold2', 'Parameters')
    min_area = cv2.getTrackbarPos('Min Area', 'Parameters')
    
    # Tìm các cạnh trong hình ảnh
    edges = cv2.Canny(gray, thresh1, thresh2)
    
    # Tìm các đường viền trong hình ảnh
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    if valid_contours:
        # Tìm đường viền lớn nhất
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # Nhận diện hình dạng của đường viền lớn nhất
        shape = detect_shape(largest_contour)
        
        # Vẽ đường viền và tên của hình dạng lên khung hình
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(frame, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    # Hiển thị số lượng đường viền hợp lệ
    cv2.putText(frame, f'Contours: {len(valid_contours)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Hiển thị khung hình
    cv2.imshow('Shape Detection', frame)
    
    # Hiển thị khung hình cạnh
    cv2.imshow('Edges', edges)
    
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()
