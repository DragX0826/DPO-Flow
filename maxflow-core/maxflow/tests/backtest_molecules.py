import torch
import numpy as np
from maxflow.models.backbone import CrossGVP
from maxflow.models.flow_matching import RectifiedFlow
from torch_geometric.data import Data, Batch
from maxflow.utils.metrics import MultiObjectiveScorer, get_mol_from_data

class MolecularBacktester:
    """
    Financial-Grade backtesting for 3D Molecule Generation.
    Inspired by Citadel/Two Sigma signal evaluation suites.
    """
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.scorer = MultiObjectiveScorer()
        
    def run_monte_carlo(self, data, num_simulations=50, steps=20):
        """
        Run multiple trajectories for the same target to evaluate stability.
        (Analogous to testing a trading strategy against 100 market realizations)
        """
        self.model.eval()
        rewards = []
        clashes = []
        
        print(f"Starting Monte Carlo Simulation (N={num_simulations})...")
        
        for i in range(num_simulations):
            with torch.no_grad():
                # Each sample starts from different noise (x_0)
                x_final, _ = self.model.sample(data.to(self.device), steps=steps)
                
                # Evaluation
                mock_data = data.clone()
                mock_data.pos_L = x_final
                mol = get_mol_from_data(mock_data)
                
                reward = self.scorer.calculate_reward(mol, x_final)
                clash = self.scorer.check_clashes(x_final)
                
                rewards.append(reward)
                clashes.append(clash)
                
        # Quant Metrics
        mean_r = np.mean(rewards)
        std_r = np.std(rewards)
        sharpe = mean_r / (std_r + 1e-6)
        max_drawdown = np.max(clashes) # Highest penalty = Worst case scenario
        
        print("\n--- Financial-Grade Backtest Results ---")
        print(f"Mean Generation Reward: {mean_r:.4f}")
        print(f"Reward Volatility (Std): {std_r:.4f}")
        print(f"Generation Sharpe Ratio: {sharpe:.4f}")
        print(f"Maximum Chemical Drawdown (Max Clashes): {max_drawdown}")
        print("----------------------------------------\n")
        
        return {
            'sharpe': sharpe,
            'mean_reward': mean_r,
            'max_drawdown': max_drawdown
        }

def backtest_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init Model
    backbone = CrossGVP(node_in_dim=161, hidden_dim=64).to(device)
    rf = RectifiedFlow(backbone).to(device)
    
    # Mock "Zero Scaffold" Target
    num_nodes = 15
    data = Data(
        x_L=torch.randn(num_nodes, 161),
        pos_L=torch.randn(num_nodes, 3), # Target (though not used in sample)
        x_P=torch.randn(100, 21),
        pos_P=torch.randn(100, 3),
        pocket_center=torch.zeros(3),
        atom_types=torch.randint(0, 10, (num_nodes,))
    )
    
    tester = MolecularBacktester(rf, device=device)
    results = tester.run_monte_carlo(data, num_simulations=10)
    
    if results['sharpe'] > 0:
        print("✅ Backtest PASSED: Strategy shows positive risk-adjusted expectation.")
    else:
        print("⚠️ Backtest WARNING: High volatility in generation.")

if __name__ == "__main__":
    backtest_demo()
