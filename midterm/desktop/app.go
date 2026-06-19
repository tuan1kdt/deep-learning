package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// App quản lý vòng đời sidecar Python và là proxy mỏng giữa frontend và sidecar.
type App struct {
	ctx     context.Context
	baseURL string // http://127.0.0.1:<port>
	cmd     *exec.Cmd
	client  *http.Client
}

func NewApp() *App {
	return &App{client: &http.Client{Timeout: 120 * time.Second}}
}

type HealthResp struct {
	Ready        bool   `json:"ready"`
	Checkpoint   string `json:"checkpoint"`
	HasAttention bool   `json:"has_attention"`
}

type CheckpointsResp struct {
	Checkpoints []string `json:"checkpoints"`
	Current     string   `json:"current"`
}

type Answer struct {
	Answer string  `json:"answer"`
	Prob   float64 `json:"prob"`
}

type PredictResp struct {
	Answers      []Answer `json:"answers"`
	Heatmap      string   `json:"heatmap"`
	HasAttention bool     `json:"has_attention"`
}

func isDir(p string) bool {
	info, err := os.Stat(p)
	return err == nil && info.IsDir()
}

// findRepoRoot đi ngược từ CWD lên trên tìm thư mục chứa cả ".venv" và "midterm".
// Override được bằng biến môi trường DEEPLEARNING_ROOT.
func findRepoRoot() (string, error) {
	if env := os.Getenv("DEEPLEARNING_ROOT"); env != "" {
		return env, nil
	}
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if isDir(filepath.Join(dir, ".venv")) && isDir(filepath.Join(dir, "midterm")) {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("không tìm thấy repo root (.venv + midterm); đặt biến DEEPLEARNING_ROOT")
		}
		dir = parent
	}
}

func freePort() (int, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port, nil
}

func (a *App) startup(ctx context.Context) {
	a.ctx = ctx
	root, err := findRepoRoot()
	if err != nil {
		fmt.Println("startup error:", err)
		return
	}
	port, err := freePort()
	if err != nil {
		fmt.Println("startup error:", err)
		return
	}
	a.baseURL = fmt.Sprintf("http://127.0.0.1:%d", port)
	python := filepath.Join(root, ".venv", "bin", "python")
	a.cmd = exec.Command(python, "-m", "midterm.serve", "--port", fmt.Sprint(port),
		"--checkpoint", "cross_attention")
	a.cmd.Dir = root
	a.cmd.Stdout = os.Stdout
	a.cmd.Stderr = os.Stderr
	if err := a.cmd.Start(); err != nil {
		fmt.Println("không khởi động được sidecar:", err)
	}
}

func (a *App) shutdown(ctx context.Context) {
	if a.cmd != nil && a.cmd.Process != nil {
		// Deliberate hard kill of the sidecar: OnShutdown fires before the Go
		// process exits, so the OS will reap the child cleanly — no zombie risk.
		_ = a.cmd.Process.Kill()
	}
}

func (a *App) getJSON(path string, out any) error {
	resp, err := a.client.Get(a.baseURL + path)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("GET %s trả về %d: %s", path, resp.StatusCode, string(b))
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func (a *App) Health() (HealthResp, error) {
	var h HealthResp
	err := a.getJSON("/health", &h)
	return h, err
}

func (a *App) Checkpoints() (CheckpointsResp, error) {
	var c CheckpointsResp
	err := a.getJSON("/checkpoints", &c)
	return c, err
}

func (a *App) LoadCheckpoint(name string) (HealthResp, error) {
	var h HealthResp
	body, _ := json.Marshal(map[string]string{"checkpoint": name})
	resp, err := a.client.Post(a.baseURL+"/load", "application/json", bytes.NewReader(body))
	if err != nil {
		return h, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return h, fmt.Errorf("load lỗi %d: %s", resp.StatusCode, string(b))
	}
	err = json.NewDecoder(resp.Body).Decode(&h)
	return h, err
}

func (a *App) Predict(imageB64 string, question string) (PredictResp, error) {
	var pr PredictResp
	raw, err := base64.StdEncoding.DecodeString(imageB64)
	if err != nil {
		return pr, fmt.Errorf("ảnh base64 không hợp lệ: %w", err)
	}
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	fw, err := w.CreateFormFile("image", "upload.png")
	if err != nil {
		return pr, err
	}
	if _, err := fw.Write(raw); err != nil {
		return pr, err
	}
	if err := w.WriteField("question", question); err != nil {
		return pr, err
	}
	w.Close()

	resp, err := a.client.Post(a.baseURL+"/predict", w.FormDataContentType(), &buf)
	if err != nil {
		return pr, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return pr, fmt.Errorf("predict lỗi %d: %s", resp.StatusCode, string(b))
	}
	err = json.NewDecoder(resp.Body).Decode(&pr)
	return pr, err
}
