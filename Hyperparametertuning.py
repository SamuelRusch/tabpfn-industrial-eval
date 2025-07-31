import torch
from torch import optim
from sklearn.model_selection import ParameterGrid
from datetime import datetime
import plotly.graph_objects as go


from tab_transformer_pytorch import FTTransformer
from RMSELoss import RMSELoss
from trainval import train_loader, val_loader
from one_epoch import train_one_epoch

def run_experiments(param_grid, train_loader, val_loader, epochs=1000):
    """
    Runs a grid of hyperparameters. For each combination:
      - trains for `epochs`
      - collects per-epoch train/val losses
      - plots & saves a histogram of those losses
      - returns a list of dicts with params & metrics
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    all_results = []

    for params in ParameterGrid(param_grid):
        print(f"\n>>> Starting run with params: {params}")

        # 1) Build fresh model, optimizer, criterion
        model = FTTransformer(
            categories=(1,),
            num_continuous=8,
            dim=params["dim"],
            dim_out=1,
            depth=params["depth"],
            heads=params["heads"],
            attn_dropout=params["attn_dropout"],
            ff_dropout=params["ff_dropout"],
        )
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )
        criterion = RMSELoss()

        train_losses = []
        val_losses   = []
        best_vloss   = float("inf")
        best_epoch   = -1

        # 2) Standard epoch loop
        for epoch in range(1, epochs+1):
            # — Training —
            model.train()
            avg_train = train_one_epoch(train_loader, model, optimizer, criterion)
            train_losses.append(avg_train)

            # — Validation —
            model.eval()
            total_val = 0.0
            with torch.no_grad():
                for x_cat, x_cont, y in val_loader:
                    y = y.unsqueeze(-1) if y.dim() == 1 else y
                    pred = model(x_cat, x_cont)
                    total_val += criterion(pred, y).item()
            avg_val = total_val / len(val_loader)
            val_losses.append(avg_val)

            # — Track best —
            if avg_val < best_vloss:
                best_vloss = avg_val
                best_epoch = epoch
                # Optional: save checkpoint
                # torch.save(model.state_dict(),
                #            f"ckpt_{timestamp}_dim{params['dim']}_ep{epoch}.pt")

            if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
                print(f" Epoch {epoch:4d} — train: {avg_train:.4f}, val: {avg_val:.4f}")

        # 3) Build & save a histogram of losses
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=train_losses,
            name="Train Losses",
            opacity=0.7,
            marker_color="royalblue"
        ))
        fig.add_trace(go.Histogram(
            x=val_losses,
            name="Validation Losses",
            opacity=0.7,
            marker_color="tomato"
        ))
        fig.update_layout(
            title=f"Loss Distribution — {params}",
            barmode="overlay",
            xaxis_title="Loss",
            yaxis_title="Count",
            template="plotly_white"
        )

        # Save to file (or just fig.show())
        fname = f"hist_{timestamp}_dim{params['dim']}_depth{params['depth']}.png"
        fig.write_image(fname)
        print(f"  ▶ Histogram saved to {fname}")

        # 4) Store results
        all_results.append({
            "params":       params,
            "train_losses": train_losses,
            "val_losses":   val_losses,
            "best_val":     best_vloss,
            "best_epoch":   best_epoch,
            "hist_path":    fname
        })

    return all_results