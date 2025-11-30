import numpy as np
import matplotlib.pyplot as plt

class LDIW_PSO:
    def __init__(self, func, n_dim, n_particles, max_iter, bounds):
        self.func = func             # Hàm mục tiêu
        self.n_dim = n_dim           # Số chiều không gian
        self.n_particles = n_particles # Số lượng hạt
        self.max_iter = max_iter     # Tổng số vòng lặp
        self.bounds = bounds         # Giới hạn tìm kiếm [min, max]
        
        # --- CẤU HÌNH TRỌNG SỐ ---
        self.w_max = 0.9             # Quán tính lớn nhất (lúc đầu)
        self.w_min = 0.4             # Quán tính nhỏ nhất (lúc cuối)
        self.c1 = 2.0                # Hệ số học tập cá nhân
        self.c2 = 2.0                # Hệ số học tập xã hội
        
        # --- KHỞI TẠO BẦY ĐÀN ---
        self.lb, self.ub = bounds
        # Vị trí ngẫu nhiên
        self.X = np.random.uniform(self.lb, self.ub, (n_particles, n_dim))
        # Vận tốc ngẫu nhiên (thường khởi tạo nhỏ)
        self.V = np.random.uniform(-1, 1, (n_particles, n_dim))
        
        # Khởi tạo P_best (Kỷ lục cá nhân)
        self.P_best_pos = self.X.copy()
        self.P_best_val = np.array([self.func(x) for x in self.X])
        
        # Khởi tạo G_best (Kỷ lục toàn đàn)
        best_idx = np.argmin(self.P_best_val)
        self.G_best_pos = self.P_best_pos[best_idx].copy()
        self.G_best_val = self.P_best_val[best_idx]
        
        # Lưu lịch sử để vẽ đồ thị
        self.history = []

    def optimize(self):
        print(f"{'Vòng lặp':<10} | {'Trọng số w':<12} | {'G_best (Fitness)':<20}")
        print("-" * 50)

        for t in range(self.max_iter):
            # --- 1. CẬP NHẬT TRỌNG SỐ W (LDIW) ---
            # Công thức giảm tuyến tính: w = w_max - (w_max - w_min) * (t / max_iter)
            w = self.w_max - (self.w_max - self.w_min) * (t / self.max_iter)
            
            # --- 2. CẬP NHẬT VẬN TỐC & VỊ TRÍ ---
            r1 = np.random.rand(self.n_particles, self.n_dim)
            r2 = np.random.rand(self.n_particles, self.n_dim)
            
            # Cập nhật vận tốc
            self.V = (w * self.V) + \
                     (self.c1 * r1 * (self.P_best_pos - self.X)) + \
                     (self.c2 * r2 * (self.G_best_pos - self.X))
            
            # Kẹp vận tốc (Optional: giúp thuật toán ổn định hơn)
            # v_limit = 0.2 * (self.ub - self.lb)
            # self.V = np.clip(self.V, -v_limit, v_limit)

            # Cập nhật vị trí
            self.X = self.X + self.V
            
            # Xử lý biên (Nếu bay ra ngoài thì kéo lại biên)
            self.X = np.clip(self.X, self.lb, self.ub)
            
            # --- 3. ĐÁNH GIÁ ---
            # Tính giá trị hàm mục tiêu cho vị trí mới
            current_vals = np.array([self.func(x) for x in self.X])
            
            # Cập nhật P_best (Cá nhân)
            # Tạo mặt nạ boolean những hạt có giá trị mới tốt hơn cũ
            better_mask = current_vals < self.P_best_val
            self.P_best_pos[better_mask] = self.X[better_mask]
            self.P_best_val[better_mask] = current_vals[better_mask]
            
            # Cập nhật G_best (Toàn đàn)
            min_val_idx = np.argmin(self.P_best_val)
            if self.P_best_val[min_val_idx] < self.G_best_val:
                self.G_best_val = self.P_best_val[min_val_idx]
                self.G_best_pos = self.P_best_pos[min_val_idx].copy()
            
            # Lưu lịch sử
            self.history.append(self.G_best_val)
            
            # In ra màn hình kết quả từng vòng
            print(f"{t+1:<10} | {w:.4f}       | {self.G_best_val:.10f}")

        return self.G_best_pos, self.G_best_val

# --- CHẠY THỬ ---

# 1. Định nghĩa hàm mục tiêu (Hàm Sphere: f(x) = sum(x^2))
# Mục tiêu: Tìm x để f(x) -> 0
def sphere_function(x):
    return np.sum(x**2)

# 2. Thiết lập tham số
n_dim = 10           # Bài toán 10 chiều
n_particles = 30     # 30 hạt
max_iter = 50        # Chạy 50 vòng
bounds = [-100, 100] # Tìm kiếm trong vùng [-100, 100]

# 3. Khởi tạo và chạy
pso = LDIW_PSO(sphere_function, n_dim, n_particles, max_iter, bounds)
best_pos, best_val = pso.optimize()

print("\n" + "="*30)
print(f"Kết quả cuối cùng:")
print(f"Giá trị nhỏ nhất (Min Fitness): {best_val}")
# print(f"Vị trí tối ưu: {best_pos}")