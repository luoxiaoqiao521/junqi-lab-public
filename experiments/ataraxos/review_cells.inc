API int review_cells(const Match* match,uint8_t* out) {
  const auto& board=match->boards[match->ply%kBuffer];
  for (int n=0;n<100;++n) {
    const auto& p=board.pieces[n/10][n%10];
    out[n]=p.color==1 || p.color==2 ? uint8_t((p.color<<4)|p.type|(p.visible?64:0)) : 0;
  }
  return match->ply;
}
