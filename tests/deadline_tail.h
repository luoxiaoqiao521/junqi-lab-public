#pragma once

struct TailWindow {
  int at=0, actor=jq::kNone, proposed=0, original=0, applied=0;
  int first_result=-1, partner_offset=0, partner_contact=0;
  int contacts=0, friendly_exit=0, enemy_exit=0, quiet_peak=0;
  uint64_t prefix=0;

  void offer(int ply,int seat,jq::Move base,jq::Move alternative,uint64_t trace) {
    if (at) return;
    at=ply+1;
    actor=seat;
    original=int(base.from)*jq::kNodes+base.to;
    proposed=int(alternative.from)*jq::kNodes+alternative.to;
    prefix=trace;
  }

  void observe(const jq::StepInfo& event,int ply,int clock) {
    if (!at || ply<at || ply>at+12) return;
    if (ply==at) first_result=static_cast<int>(event.outcome);
    const bool contact=event.outcome!=jq::Outcome::Move;
    contacts+=contact;
    quiet_peak=std::max(quiet_peak,clock);
    if (!partner_offset && event.seat==(actor+2)%jq::kSeats) {
      partner_offset=ply-at;
      partner_contact=contact;
    }
    for(int seat=0;seat<jq::kSeats;++seat) if (event.eliminated_mask&(1u<<seat)) {
      if (jq::team_of(seat)==jq::team_of(actor)) ++friendly_exit;
      else ++enemy_exit;
    }
  }

  void write(std::ostream& out) const {
    out<<' '<<at<<' '<<actor<<' '<<original<<' '<<proposed<<' '<<applied<<' '<<prefix<<' '<<first_result
       <<' '<<partner_offset<<' '<<partner_contact<<' '<<contacts<<' '<<friendly_exit<<' '<<enemy_exit<<' '<<quiet_peak;
  }
};
