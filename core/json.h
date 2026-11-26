// Small bounded JSON reader for the local configuration/log schemas (integer numbers).
#pragma once
#include <charconv>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
namespace jq {
struct Json {
  char type = 'n';
  std::string text;
  std::map<std::string, Json> object;
  std::vector<Json> array;
  const Json& at(const std::string& key) const {
    if (type != 'o' || !object.count(key)) throw std::runtime_error("missing JSON key: " + key);
    return object.at(key);
  }
  std::string str() const {
    if (type != 's') throw std::runtime_error("expected string");
    return text;
  }
  int integer() const {
    int n = 0;
    auto v = std::from_chars(text.data(), text.data() + text.size(), n);
    if (type != 'i' || v.ec != std::errc{} || v.ptr != text.data() + text.size())
      throw std::runtime_error("expected bounded integer");
    return n;
  }
  bool boolean() const {
    if (type != 'b') throw std::runtime_error("expected boolean");
    return text == "true";
  }
};
class JsonReader {
  const std::string& s; size_t p = 0;
  char peek() const { return p < s.size() ? s[p] : '\0'; }
  void ws() { while (peek() == ' ' || peek() == '\n' || peek() == '\r' || peek() == '\t') ++p; }
  void expect(char c) { ws(); if (peek() != c) throw std::runtime_error("invalid JSON delimiter"); ++p; }
  unsigned unicode_unit() {
    if (s.size()-p<4) throw std::runtime_error("incomplete Unicode escape");
    unsigned value=0;
    const char* end=s.data()+p+4;
    const auto parsed=std::from_chars(s.data()+p,end,value,16);
    if (parsed.ec!=std::errc{} || parsed.ptr!=end) throw std::runtime_error("invalid Unicode escape");
    p+=4;
    return value;
  }
  void append_unicode(std::string& out) {
    unsigned point=unicode_unit();
    if (point>=0xd800 && point<=0xdbff) {
      if (s.compare(p,2,"\\u")!=0) throw std::runtime_error("missing low surrogate");
      p+=2;
      const unsigned low=unicode_unit();
      if (low<0xdc00 || low>0xdfff) throw std::runtime_error("invalid low surrogate");
      point=0x10000+((point-0xd800)<<10)+(low-0xdc00);
    } else if (point>=0xdc00 && point<=0xdfff) {
      throw std::runtime_error("unpaired low surrogate");
    }
    if (point<0x80) {
      out+=char(point);
    } else {
      if (point>=0x10000) out+=char(0xf0|(point>>18));
      if (point>=0x800) out+=char((point<0x10000?0xe0:0x80)|((point>>12)&0x3f));
      out+=char((point<0x800?0xc0:0x80)|((point>>6)&0x3f));
      out+=char(0x80|(point&0x3f));
    }
  }
  std::string string() {
    expect('"'); std::string out;
    while (peek() && peek() != '"') {
      unsigned char c = static_cast<unsigned char>(s[p++]);
      if (c < 32) throw std::runtime_error("control in string");
      if (c != '\\') { out += static_cast<char>(c); continue; }
      char e = peek(); if (!e) throw std::runtime_error("unfinished escape"); ++p;
      if (e == '"' || e == '\\' || e == '/') out += e;
      else if (e == 'n') out += '\n'; else if (e == 'r') out += '\r';
      else if (e == 't') out += '\t'; else if (e == 'b') out += '\b';
      else if (e == 'f') out += '\f';
      else if (e == 'u') append_unicode(out);
      else throw std::runtime_error("unsupported JSON escape");
    }
    expect('"'); return out;
  }
  Json value(int depth) {
    if (depth > 20) throw std::runtime_error("JSON nesting limit");
    ws(); Json v; char c = peek();
    if (c == '"') { v.type = 's'; v.text = string(); return v; }
    if (c == '{' || c == '[') {
      ++p; v.type = c == '{' ? 'o' : 'a'; ws(); char end = c == '{' ? '}' : ']';
      if (peek() == end) { ++p; return v; }
      while (true) {
        if (c == '{') {
          std::string k = string(); expect(':');
          if (v.object.count(k)) throw std::runtime_error("duplicate JSON key");
          v.object.emplace(k, value(depth + 1));
        } else v.array.push_back(value(depth + 1));
        ws(); if (peek() == end) { ++p; return v; } expect(',');
      }
    }
    size_t start = p;
    while (peek() && peek() != ',' && peek() != '}' && peek() != ']' && peek() != ' ' && peek() != '\n' && peek() != '\r' && peek() != '\t') ++p;
    v.text = s.substr(start, p - start);
    if (v.text == "true" || v.text == "false") v.type = 'b';
    else if (v.text == "null") v.type = 'n';
    else {
      v.type = 'i'; v.integer();
      size_t first = v.text[0] == '-' ? 1 : 0;
      if (v.text[first] == '0' && v.text.size() > first + 1) throw std::runtime_error("leading zero");
    }
    return v;
  }
 public:
  explicit JsonReader(const std::string& input) : s(input) {}
  Json parse() {
    if (s.size() > 16000000) throw std::runtime_error("JSON size limit");
    Json v = value(0); ws(); if (p != s.size()) throw std::runtime_error("JSON trailing data"); return v;
  }
};
inline Json parse_json(const std::string& s) { return JsonReader(s).parse(); }
inline std::string quote_json(const std::string& s) {
  std::string out = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
    else if (c == '\n') out += "\\n"; else if (c == '\r') out += "\\r";
    else if (c == '\t') out += "\\t";
    else if (c == '\b') out += "\\b"; else if (c == '\f') out += "\\f";
    else if (c < 32) {
      const char* hex = "0123456789abcdef";
      out += "\\u00"; out += hex[c>>4]; out += hex[c&15];
    }
    else out += char(c);
  }
  return out + '"';
}
inline std::string dump_json(const Json& v) {
  if (v.type == 's') return quote_json(v.text);
  if (v.type == 'n') return "null";
  if (v.type != 'a' && v.type != 'o') return v.text;
  std::string out = v.type == 'a' ? "[" : "{"; bool first = true;
  if (v.type == 'a') for (const auto& a : v.array) {
    if (!first) out += ','; first = false; out += dump_json(a);
  }
  else for (const auto& kv : v.object) {
    if (!first) out += ','; first = false; out += quote_json(kv.first) + ':' + dump_json(kv.second);
  }
  return out + (v.type == 'a' ? ']' : '}');
}

}
