import numpy as np
import matplotlib.pyplot as plt

class TVAC_PSO:
    def __init__(self, func, n_dim, n_particles, max_iter, bounds):
        self.func = func             # Hàm mục tiêu
        self.n_dim = n_dim           # Số chiều
        self.n_particles = n_particles # Số lượng hạt
        self.max_iter = max_iter     # Tổng số vòng lặp
        self.bounds = bounds         # Giới hạn tìm kiếm [min, max]
        
        # --- Cấu hình TVAC (Time-Varying Acceleration Coefficients) ---
        # w: Quán tính (Giảm dần để chuyển từ Thăm dò -> Khai thác)
        self.w_max = 0.9
        self.w_min = 0.4
        
        # c1: Cá nhân (Giảm dần - Bớt tin vào bản thân về sau)
        self.c1_start = 2.5
        self.c1_end   = 0.5
        
        # c2: Xã hội (Tăng dần - Tin vào tập thể nhiều hơn về sau)
        self.c2_start = 0.5
        self.c2_end   = 2.5
        
        # --- Khởi tạo Bầy đàn ---
        self.lb, self.ub = bounds
        self.X = np.random.uniform(self.lb, self.ub, (n_particles, n_dim))
        self.V = np.random.uniform(-1, 1, (n_particles, n_dim))
        
        # Khởi tạo P_best (Cá nhân tốt nhất)
        self.P_best_pos = self.X.copy()
        self.P_best_val = np.array([self.func(x) for x in self.X])
        
        # Khởi tạo G_best (Toàn đàn tốt nhất)
        best_idx = np.argmin(self.P_best_val)
        self.G_best_pos = self.P_best_pos[best_idx].copy()
        self.G_best_val = self.P_best_val[best_idx]
        
        # Lưu lịch sử để vẽ đồ thị
        self.history = []

    def optimize(self):
        print(f"{'Iter':<5} | {'Best Fitness':<15} | {'w':<6} | {'c1':<6} | {'c2':<6}")
        print("-" * 55)

        for t in range(self.max_iter):
            # --- 1. TÍNH TOÁN THAM SỐ ĐỘNG (TVAC LOGIC) ---
            # Tỷ lệ hoàn thành (từ 0 đến 1)
            fraction = t / self.max_iter
            
            # w giảm tuyến tính
            w = (self.w_max - self.w_min) * (1 - fraction) + self.w_min
            
            # c1 giảm tuyến tính
            c1 = (self.c1_start - self.c1_end) * (1 - fraction) + self.c1_end
            
            # c2 tăng tuyến tính
            c2 = (self.c2_start - self.c2_end) * fraction + self.c2_start
            
            # --- 2. CẬP NHẬT TRẠNG THÁI ---
            r1 = np.random.rand(self.n_particles, self.n_dim)
            r2 = np.random.rand(self.n_particles, self.n_dim)
            
            # Cập nhật vận tốc
            self.V = (w * self.V) + \
                     (c1 * r1 * (self.P_best_pos - self.X)) + \
                     (c2 * r2 * (self.G_best_pos - self.X))
            
            # Kẹp vận tốc (Optional: Giới hạn V_max giúp ổn định hơn)
            v_max = 0.2 * (self.ub - self.lb)
            self.V = np.clip(self.V, -v_max, v_max)

            # Cập nhật vị trí
            self.X = self.X + self.V
            
            # Kẹp vị trí (Không cho bay ra khỏi biên)
            self.X = np.clip(self.X, self.lb, self.ub)
            
            # --- 3. ĐÁNH GIÁ ---
            # Tính fitness
            current_vals = np.array([self.func(x) for x in self.X])
            
            # Cập nhật P_best
            better_mask = current_vals < self.P_best_val
            self.P_best_pos[better_mask] = self.X[better_mask]
            self.P_best_val[better_mask] = current_vals[better_mask]
            
            # Cập nhật G_best
            min_val_idx = np.argmin(self.P_best_val)
            if self.P_best_val[min_val_idx] < self.G_best_val:
                self.G_best_val = self.P_best_val[min_val_idx]
                self.G_best_pos = self.P_best_pos[min_val_idx].copy()
            
            # Lưu và In kết quả
            self.history.append(self.G_best_val)
            
            # In ra mỗi 5 vòng lặp (hoặc mỗi vòng nếu muốn)
            if t % 1 == 0: 
                print(f"{t+1:<5} | {self.G_best_val:.8f}      | {w:.3f}  | {c1:.3f}  | {c2:.3f}")

        return self.G_best_pos, self.G_best_val


# 1. Định nghĩa hàm mục tiêu: Sphere Function f(x) = x^2 + y^2 + ...
# Giá trị tối ưu là 0 tại vị trí [0, 0, ..., 0]
def sphere_function(x):
    return np.sum(x**2)

# 2. Cấu hình bài toán
n_dim = 10          # Bài toán 10 chiều
n_particles = 30    # 30 hạt
max_iter = 100       # 50 vòng lặp
bounds = [-100, 100] # Tìm kiếm trong khoảng -100 đến 100

# 3. Khởi tạo và chạy TVAC-PSO
print("BẮT ĐẦU CHẠY TVAC-PSO CHO HÀM SPHERE (10 CHIỀU)")
pso_solver = TVAC_PSO(sphere_function, n_dim, n_particles, max_iter, bounds)
best_pos, best_val = pso_solver.optimize()

print("\n" + "="*30)
print(f"KẾT QUẢ CUỐI CÙNG:")
print(f"Giá trị nhỏ nhất tìm được: {best_val}")
# print(f"Vị trí tối ưu: {best_pos}") # Bỏ comment nếu muốn xem tọa độ