const API_BASE = 'http://localhost:8000';
function fileUrl(absolutePath) {
  if (!absolutePath) return "";
  const normalizedPath = absolutePath.replace(/\\/g, '/');
  console.log("Normalized:", normalizedPath);
  
  const marker = normalizedPath.includes("/uploads/") ? "/uploads/" : "/reports/";
  console.log("Marker:", marker);
  
  const idx = normalizedPath.indexOf(marker);
  console.log("Idx:", idx);
  
  const relative = idx >= 0 ? normalizedPath.slice(idx) : normalizedPath;
  console.log("Relative:", relative);
  
  return `${API_BASE}/files${relative}`;
}

console.log("Final URL:", fileUrl("C:\\Users\\Preeti Devaraddi\\Documents\\ChestVision-AI\\backend\\reports\\report_2011d6ca97c74de89431df0464631202.pdf"));
