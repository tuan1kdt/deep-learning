package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// newTestApp trỏ baseURL vào một mock server thay cho sidecar thật.
func newTestApp(srv *httptest.Server) *App {
	a := NewApp()
	a.baseURL = srv.URL
	return a
}

func TestHealthParsesJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"ready":true,"checkpoint":"cross_attention","has_attention":true}`))
	}))
	defer srv.Close()
	a := newTestApp(srv)
	h, err := a.Health()
	if err != nil {
		t.Fatal(err)
	}
	if !h.Ready || h.Checkpoint != "cross_attention" || !h.HasAttention {
		t.Fatalf("parse sai: %+v", h)
	}
}

func TestHealthHandlesNullCheckpoint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"ready":false,"checkpoint":null,"has_attention":false}`))
	}))
	defer srv.Close()
	a := newTestApp(srv)
	h, err := a.Health()
	if err != nil {
		t.Fatal(err)
	}
	if h.Ready || h.Checkpoint != "" {
		t.Fatalf("null checkpoint phải thành \"\": %+v", h)
	}
}

func TestPredictSendsMultipartAndParses(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Errorf("không parse được multipart: %v", err)
		}
		if r.FormValue("question") != "is there cardiomegaly?" {
			t.Errorf("question sai: %q", r.FormValue("question"))
		}
		if _, _, err := r.FormFile("image"); err != nil {
			t.Errorf("thiếu file image: %v", err)
		}
		w.Write([]byte(`{"answers":[{"answer":"no","prob":0.8},{"answer":"yes","prob":0.2}],"heatmap":"QUJD","has_attention":true}`))
	}))
	defer srv.Close()
	a := newTestApp(srv)
	// "QUJD" là base64 của "ABC" → ảnh giả hợp lệ cho test
	pr, err := a.Predict("QUJD", "is there cardiomegaly?")
	if err != nil {
		t.Fatal(err)
	}
	if len(pr.Answers) != 2 || pr.Answers[0].Answer != "no" || !pr.HasAttention {
		t.Fatalf("parse sai: %+v", pr)
	}
}

func TestPredictErrorsOnNon200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(409)
		w.Write([]byte(`{"error":"model đang tải"}`))
	}))
	defer srv.Close()
	a := newTestApp(srv)
	if _, err := a.Predict("QUJD", "q"); err == nil {
		t.Fatal("mong đợi lỗi khi status != 200")
	}
}

func TestCheckpointsParsesJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"checkpoints":["concat","cross_attention"],"current":"concat"}`))
	}))
	defer srv.Close()
	a := newTestApp(srv)
	c, err := a.Checkpoints()
	if err != nil {
		t.Fatal(err)
	}
	if len(c.Checkpoints) != 2 || c.Current != "concat" {
		t.Fatalf("parse sai: %+v", c)
	}
}

func TestLoadCheckpointErrorsOnNon200(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(404)
		w.Write([]byte(`{"error":"không tìm thấy checkpoint"}`))
	}))
	defer srv.Close()
	a := newTestApp(srv)
	if _, err := a.LoadCheckpoint("khong_ton_tai"); err == nil {
		t.Fatal("mong đợi lỗi khi status != 200")
	}
}
