import numpy as np #type:ignore

class ParticleSwarmOptimization:
    def __init__(self, func, n_dim, n_particles, iterations):
        self.func = func             # Hàm mục tiêu cần tối ưu
        self.n_dim = n_dim           # Số chiều của bài toán
        self.n_particles = n_particles # Số lượng hạt
        self.iterations = iterations   # Số vòng lặp
        
        # Tham số PSO cơ bản
        self.w = 0.5    # Quán tính (Inertia)
        self.c1 = 1.5   # Hệ số cá nhân (Cognitive)
        self.c2 = 1.5   # Hệ số xã hội (Social)
        
        # Khởi tạo vị trí và vận tốc ngẫu nhiên
        # Giả sử vùng tìm kiếm từ -10 đến 10
        self.X = np.random.uniform(-10, 10, (n_particles, n_dim))
        self.V = np.random.uniform(-1, 1, (n_particles, n_dim))
        
        # Khởi tạo P_best (tốt nhất của từng hạt)
        self.P_best_pos = self.X.copy()
        self.P_best_val = np.array([func(x) for x in self.X])
        
        # Khởi tạo G_best (tốt nhất của cả đàn)
        best_idx = np.argmin(self.P_best_val)
        self.G_best_pos = self.P_best_pos[best_idx].copy()
        self.G_best_val = self.P_best_val[best_idx]

    def optimize(self):
        for i in range(self.iterations):
            # Tạo số ngẫu nhiên r1, r2
            r1 = np.random.rand(self.n_particles, self.n_dim)
            r2 = np.random.rand(self.n_particles, self.n_dim)
            
            # --- CẬP NHẬT VẬN TỐC ---
            # V_new = w*V + c1*r1*(P_best - X) + c2*r2*(G_best - X)
            self.V = (self.w * self.V) + \
                     (self.c1 * r1 * (self.P_best_pos - self.X)) + \
                     (self.c2 * r2 * (self.G_best_pos - self.X))
            
            # --- CẬP NHẬT VỊ TRÍ ---
            self.X = self.X + self.V
            
            # --- ĐÁNH GIÁ VÀ CẬP NHẬT ---
            # Tính giá trị hàm mục tiêu tại vị trí mới
            current_vals = np.array([self.func(x) for x in self.X])
            
            # Cập nhật P_best (Nếu vị trí mới tốt hơn vị trí cũ của chính nó)
            better_mask = current_vals < self.P_best_val
            self.P_best_pos[better_mask] = self.X[better_mask]
            self.P_best_val[better_mask] = current_vals[better_mask]
            
            # Cập nhật G_best (Nếu tìm thấy hạt nào tốt hơn G_best hiện tại)
            min_val_idx = np.argmin(self.P_best_val)
            if self.P_best_val[min_val_idx] < self.G_best_val:
                self.G_best_val = self.P_best_val[min_val_idx]
                self.G_best_pos = self.P_best_pos[min_val_idx].copy()
            
            # In kết quả tốt nhất sau mỗi vòng lặp
            print(f"Vòng lặp {i+1}: G_best = {self.G_best_val:.6f}, Vị trí = {self.G_best_pos}")
                
        return self.G_best_pos, self.G_best_val

# --- CHẠY THỬ ---

# Hàm mục tiêu: f(x) = tổng bình phương (Sphere function)
def sphere_function(x):
    return np.sum(x**2)

# Thiết lập: 2 chiều (x, y), 30 hạt, 50 vòng lặp
pso = ParticleSwarmOptimization(func=sphere_function, n_dim=2, n_particles=30, iterations=50)

print("Bắt đầu tối ưu hóa...")
best_pos, best_val = pso.optimize()

print("-" * 30)
print(f"Vị trí tối ưu tìm được: {best_pos}")
print(f"Giá trị nhỏ nhất: {best_val}")