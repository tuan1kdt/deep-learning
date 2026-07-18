# vivlm/tests/test_plots.py
from vivlm.plot_logs import plot_gpu, plot_pretrain, plot_sft


def test_plot_pretrain(tmp_path):
    c = tmp_path / "p.csv"
    c.write_text("step,split,loss,lr,tok_per_s,vram_gb\n"
                 "10,train,5.1,1e-4,80000,22.1\n"
                 "20,train,4.8,2e-4,81000,22.1\n"
                 "20,val,4.9,,,\n")
    out = tmp_path / "p.png"
    plot_pretrain(str(c), str(out))
    assert out.exists()


def test_plot_sft(tmp_path):
    c = tmp_path / "s.csv"
    c.write_text("phase,step,loss,lr\nprojector,1,3.2,1e-3\nfull,1,2.9,1e-4\n")
    out = tmp_path / "s.png"
    plot_sft(str(c), str(out))
    assert out.exists()


def test_plot_gpu(tmp_path):
    c = tmp_path / "g.csv"
    c.write_text("2026/07/20 10:00:00.000, 97, 22800\n"
                 "2026/07/20 10:00:30.000, 99, 23100\n")
    out = tmp_path / "g.png"
    plot_gpu(str(c), str(out))
    assert out.exists()
