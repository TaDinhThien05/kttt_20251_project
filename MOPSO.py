import numpy as np #type: ignore
import matplotlib.pyplot as plt #type:ignore

class Solution:
    def __init__(self, x, obj):
        self.x = x  # Vị trí trong không gian quyết định
        self.obj = np.array(obj)  # Vector mục tiêu [f1, f2, ...]

def check_dominance(sol1, sol2):
    """
    Kiểm tra xem sol1 có trội hơn sol2 không (Bài toán Min-Min)
    Trả về True nếu sol1 dominate sol2
    """
    # 1. sol1 phải <= sol2 ở mọi mục tiêu
    condition1 = np.all(sol1.obj <= sol2.obj)
    # 2. sol1 phải < sol2 ở ít nhất 1 mục tiêu
    condition2 = np.any(sol1.obj < sol2.obj)
    
    return condition1 and condition2

def update_archive(archive, new_solution):
    """
    Cập nhật kho lưu trữ với giải pháp mới
    """
    # Bước 1: Kiểm tra xem new_solution có bị ai trong kho dominate không
    for member in archive:
        if check_dominance(member, new_solution):
            return archive  # Bị dominate, vứt đi, không làm gì cả

    # Bước 2: Nếu không bị dominate, nó xứng đáng vào kho.
    # Nhưng trước khi vào, vứt hạt mà nó dominate ra ngoài.
    non_dominated_archive = []
    for member in archive:
        if check_dominance(new_solution, member):
            continue 
        else:
            non_dominated_archive.append(member)
            
    non_dominated_archive.append(new_solution)
    return non_dominated_archive

class MultiObjectivePSO:
    def __init__(self, func, n_dim, n_particles, iterations, bounds=(-10, 10)):
        """
        func: Hàm mục tiêu trả về vector [f1, f2, ...]
        n_dim: Số chiều của bài toán
        n_particles: Số lượng hạt
        iterations: Số vòng lặp
        bounds: Giới hạn không gian tìm kiếm (min, max)
        """
        self.func = func
        self.n_dim = n_dim
        self.n_particles = n_particles
        self.iterations = iterations
        self.bounds = bounds
        
        # Tham số PSO
        self.w = 0.5    # Quán tính (Inertia)
        self.c1 = 1.5   # Hệ số cá nhân (Cognitive)
        self.c2 = 1.5   # Hệ số xã hội (Social)
        
        # Khởi tạo vị trí và vận tốc ngẫu nhiên
        self.X = np.random.uniform(bounds[0], bounds[1], (n_particles, n_dim))
        self.V = np.random.uniform(-1, 1, (n_particles, n_dim))
        
        # Khởi tạo P_best (tốt nhất của từng hạt) - lưu dưới dạng Solution
        self.P_best = []
        for i in range(n_particles):
            obj = func(self.X[i])
            self.P_best.append(Solution(self.X[i].copy(), obj))
        
        # Khởi tạo Archive (kho lưu trữ các giải pháp Pareto)
        self.archive = []
        for p_best in self.P_best:
            if not self.archive:
                self.archive.append(p_best)
            else:
                self.archive = update_archive(self.archive, p_best)
    
    def select_leader(self):
        """
        Chọn leader từ archive (chọn ngẫu nhiên một giải pháp từ archive)
        """
        if not self.archive:
            return None
        return np.random.choice(self.archive)
    
    def optimize(self):
        """
        Chạy thuật toán MOPSO
        """
        for i in range(self.iterations):
            # Tạo số ngẫu nhiên r1, r2
            r1 = np.random.rand(self.n_particles, self.n_dim)
            r2 = np.random.rand(self.n_particles, self.n_dim)
            
            # --- CẬP NHẬT VẬN TỐC ---
            # Với mỗi particle, chọn P_best và G_best (leader từ archive)
            for j in range(self.n_particles):
                # Chọn leader từ archive
                leader = self.select_leader()
                if leader is not None:
                    # V_new = w*V + c1*r1*(P_best - X) + c2*r2*(G_best - X)
                    self.V[j] = (self.w * self.V[j]) + \
                               (self.c1 * r1[j] * (self.P_best[j].x - self.X[j])) + \
                               (self.c2 * r2[j] * (leader.x - self.X[j]))
                else:
                    # Nếu không có leader, chỉ dùng P_best
                    self.V[j] = (self.w * self.V[j]) + \
                               (self.c1 * r1[j] * (self.P_best[j].x - self.X[j]))
            
            # --- CẬP NHẬT VỊ TRÍ ---
            self.X = self.X + self.V
            
            # Giới hạn vị trí trong bounds
            self.X = np.clip(self.X, self.bounds[0], self.bounds[1])
            
            # --- ĐÁNH GIÁ VÀ CẬP NHẬT ---
            # Tính giá trị hàm mục tiêu tại vị trí mới
            for j in range(self.n_particles):
                current_obj = self.func(self.X[j])
                current_sol = Solution(self.X[j].copy(), current_obj)
                
                # Cập nhật P_best (Nếu vị trí mới dominate hoặc không bị dominate bởi P_best cũ)
                if check_dominance(current_sol, self.P_best[j]):
                    self.P_best[j] = current_sol
                elif not check_dominance(self.P_best[j], current_sol):
                    # Nếu không ai dominate ai, chọn ngẫu nhiên hoặc giữ nguyên
                    # Ở đây ta giữ nguyên P_best
                    pass
                
                # Cập nhật Archive
                self.archive = update_archive(self.archive, current_sol)
                
        return self.archive

# --- HÀM MỤC TIÊU VÍ DỤ ---
def schaffer_function(x):
    """
    f1 = tổng x^2 
    f2 = tổng (x-2)^2 
    """
    f1 = np.sum(x**2)
    f2 = np.sum((x - 2)**2)
    return [f1, f2]

if __name__ == "__main__":
    print("Bắt đầu tối ưu hóa đa mục tiêu với MOPSO...")
    print("=" * 60)
    
    # Thiết lập: 1 chiều, 30 hạt, 50 vòng lặp
    mopso = MultiObjectivePSO(
        func=schaffer_function,
        n_dim=4,
        n_particles=30,
        iterations=50,
        bounds=(-1, 3)
    )
    
    archive = mopso.optimize()
    
    print("=" * 60)
    print(f"Kết thúc! Tìm được {len(archive)} giải pháp Pareto")
    print("\nTất cả các giải pháp Pareto:")
    for idx, sol in enumerate(archive):
        print(f"  {idx+1}. x = {sol.x}, f1 = {sol.obj[0]:.6f}, f2 = {sol.obj[1]:.6f}")
