use actix_web::http::header;
use actix_web::web::Json;
use actix_web::{get, post, HttpResponse};
use core::str;
use image::{ImageBuffer, Rgba};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::{char, fs, string};

#[derive(Deserialize)]
struct GenerateGraphRequest {
    width: usize,
    height: usize,
    pr_of_break_wall: usize,
    num_of_finish_edge: usize,
}

#[derive(Serialize)]
struct GenerateGraphResponse {
    graph: String,
}
#[derive(Serialize, Deserialize)]
struct NodeData {
    value: usize,
    isFinal: bool,
    neighbors: Vec<NeighborData>,
}
#[derive(Serialize, Deserialize)]
struct GraphData {
    nodes: Vec<NodeData>,
}
#[derive(Serialize, Deserialize)]
struct NeighborData {
    neighbor: usize,
    symbol: char,
}
struct Node {
    value: usize,
    isFinal: bool,
    neighbors: Vec<(usize, char)>,
}
struct point {
    x: usize,
    y: usize,
}
// Define a struct to represent the graph
pub struct Graph {
    nodes: HashMap<usize, Node>,
}
impl Clone for Node {
    fn clone(&self) -> Self {
        Node {
            value: self.value,
            isFinal: self.isFinal,
            neighbors: self.neighbors.clone(),
        }
    }
}
impl Clone for point {
    fn clone(&self) -> Self {
        point {
            x: self.x,
            y: self.y,
        }
    }
}

impl Graph {
    // Create a new graph
    pub fn new() -> Self {
        Graph {
            nodes: HashMap::new(),
        }
    }

    // Add a new node to the graph
    fn add_node(&mut self, value: usize, isFinal: bool) -> usize {
        let index = value;
        self.nodes.insert(
            value,
            Node {
                value,
                isFinal,
                neighbors: Vec::new(),
            },
        );
        value
    }

    // Add an edge between two nodes in the graph
    fn add_edge(&mut self, from: usize, to: usize, symbol: char) {
        if self.nodes.get(&from).is_some() {
            if let Some(node) = self.nodes.get_mut(&from) {
                if !node.neighbors.contains(&(to, symbol)) {
                    node.neighbors.push((to, symbol));
                }
            }
        }
    }
    fn has_edge(&self, from: usize, to: usize) -> bool {
        if let Some(node) = self.nodes.get(&from) {
            node.neighbors.iter().any(|&(neighbor, _)| neighbor == to)
        } else {
            false
        }
    }

    // Print out the adjacency list representation of the graph
    fn print_adjacency_list(&self) {
        for (key, node) in self.nodes.iter() {
            println!("Node {}: {:?}", key, node.value);
            println!("Neighbors: {:?}", node.neighbors);
        }
    }

    pub fn to_json_string(&self) -> String {
        let nodes: Vec<String> = self
            .nodes
            .iter()
            .map(|(node_key, node)| {
                let neighbors: Vec<String> = node
                    .neighbors
                    .iter()
                    .map(|&(neighbor, symbol)| {
                        format!(r#"{{"neighbor":{}, "symbol":"{}"}}"#, neighbor, symbol)
                    })
                    .collect();

                format!(
                    r#"{{"id":{}, "value":{}, "isFinal":{}, "neighbors":[{}]}}"#,
                    node_key,
                    node.value,
                    node.isFinal,
                    neighbors.join(", ")
                )
            })
            .collect();

        format!(r#"{{"nodes":[{}]}}"#, nodes.join(", "))
    }

    // Вывод графа в формате JSON
    fn print_json(&self) {
        println!("{}", self.to_json_string());
    }

    fn print_json_to_file(&self, filename: &str) -> std::io::Result<()> {
        let json_string = self.to_json_string();
        fs::write(filename, json_string)
    }

    pub fn from_json_string(json: &str) -> Result<Self, serde_json::Error> {
        let graph_data: GraphData = serde_json::from_str(json)?;
        let node_data = graph_data.nodes;
        let mut graph = Graph::new();

        // Добавляем узлы и ребра в граф
        for node in node_data {
            let index = graph.add_node(node.value, node.isFinal);
            for neighbor in node.neighbors {
                graph.add_edge(index, neighbor.neighbor, neighbor.symbol);
            }
        }

        Ok(graph)
    }

    pub fn from_json(graph_data: &GraphData) -> Result<Self, serde_json::Error> {
        let node_data = &graph_data.nodes;
        let mut graph = Graph::new();

        // Добавляем узлы и ребра в граф
        for node in node_data {
            let index = graph.add_node(node.value, node.isFinal);
            for neighbor in &node.neighbors {
                graph.add_edge(index, neighbor.neighbor, neighbor.symbol);
            }
        }

        Ok(graph)
    }
    pub fn complete_with_alphabet(&mut self, alphabet: Vec<char>) {
        for node in self.nodes.values_mut() {
            for symbol in alphabet.iter() {
                // Проверяем, есть ли переход по символу, если нет, добавляем самопереход
                if !node.neighbors.iter().any(|&(_, c)| c == *symbol) {
                    node.neighbors.push((node.value, *symbol));
                }
            }
        }
    }

    fn get_reverse_path(&self) -> HashMap<(usize, char), Vec<usize>> {
        let mut reverse_path: HashMap<(usize, char), Vec<usize>> = HashMap::new();
        for (vertex, node) in self.nodes.iter() {
            for (neighbor, edge_char) in node.neighbors.iter() {
                reverse_path
                    .entry((*neighbor, *edge_char))
                    .or_insert(Vec::new())
                    .push(*vertex);
            }
        }
        reverse_path
    }
    fn can_traversal(&self, start_node_index: usize) -> Vec<usize> {
        let mut result = Vec::new();
        let mut used = vec![false; self.nodes.len()];
        let mut queue = VecDeque::new();

        queue.push_back(start_node_index);
        used[start_node_index] = true;

        while let Some(node_index) = queue.pop_front() {
            result.push(node_index);

            if let Some(node) = self.nodes.get(&node_index) {
                for &(neighbor_index, _) in &node.neighbors {
                    if !used[neighbor_index] {
                        queue.push_back(neighbor_index);
                        used[neighbor_index] = true;
                    }
                }
            }
        }

        result
    }
    pub fn travers(&self, start_node_index: usize, path: String) -> bool {
        let word: Vec<char> = path.chars().collect();

        let mut current_node_index = start_node_index;
        for symbol in word.iter() {
            let next_node_index = self.nodes[&current_node_index]
                .neighbors
                .iter()
                .find(|&&(neighbor, c)| c == *symbol)
                .map(|&(neighbor, _)| neighbor);
            match next_node_index {
                Some(node_index) => current_node_index = node_index,
                None => return false,
            }
        }

        self.nodes[&current_node_index].isFinal
    }
    fn visualize_travers(
        &self,
        start_node_index: usize,
        width: usize,
        height: usize,
        path: String,
        filename: &str,
    ) -> bool {
        true
    }
    fn print_dot(&self) {
        println!("digraph G {{");
        for (i, node) in &self.nodes {
            println!("  {} [label=\"{}\"];", i, node.value);
            for &(neighbor, symbol) in &node.neighbors {
                println!("  {} -> {} [label=\"{}\"];", i, neighbor, symbol);
            }
        }
        println!("}}");
    }
    fn print_svg_rect(&self, width: usize, height: usize) {
        let _width = width * 2 + 1;
        let _height = height * 2 + 1;
        let mut svg = String::new();
        svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?><svg width=\"");
        svg.push_str(&_width.to_string());
        svg.push_str("\" height=\"");
        svg.push_str(&_height.to_string());
        svg.push_str("\">\n");
        println!("{}", svg);
        let mut table = vec![vec![false; _width]; _height];

        for (i, node) in &self.nodes {
            let index = node.value as usize - 1; // Предполагаем, что узлы нумеруются с 1
            let row = (index / width) * 2 + 1;
            let col = (index % width) * 2 + 1;

            // Добавляем ребра
            table[row][col] = true;
            for &(neighbor_index, symbol) in &node.neighbors {
                /* N
                W     E
                S
                */
                if symbol == 'W' {
                    // Вправо
                    table[row][col - 1] = true;
                } else if symbol == 'E' {
                    // Влево
                    table[row][col + 1] = true;
                } else if symbol == 'S' {
                    // Вниз
                    table[row - 1][col] = true;
                } else if symbol == 'N' {
                    // Вверх
                    table[row + 1][col] = true;
                }
            }
        }

        // Добавляем стены
        for i in 0.._height {
            for j in 0.._width {
                let mut svg = String::new();
                if table[i][j] == true {
                    svg.push_str("  <rect x=\"");
                    svg.push_str(&j.to_string());

                    svg.push_str("\" y=\"");
                    svg.push_str(&i.to_string());

                    svg.push_str("\" width=\"1\" height=\"1\" fill=\"white\"/>\n");
                } else {
                    svg.push_str("  <rect x=\"");
                    svg.push_str(&j.to_string());

                    svg.push_str("\" y=\"");
                    svg.push_str(&i.to_string());

                    svg.push_str("\" width=\"1\" height=\"1\" fill=\"black\"/>");
                }
                println!("{}", svg);
            }
        }
        println!("<rect x=\"1\" y=\"1\" width=\"1\" height=\"1\" fill=\"green\"/></svg>");
    }
    fn print_png_rect(&self, width: usize, height: usize, filename: &str) {
        let _width = (width * 2 + 1) as u32;
        let _height = (height * 2 + 1) as u32;

        // Создаем новый буфер изображения
        let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(_width, _height);

        // Заполняем изображение белым цветом (фон)
        for y in 0.._height {
            for x in 0.._width {
                img.put_pixel(x, y, Rgba([255, 255, 0, 255])); // Белый цвет
            }
        }

        // Создаем таблицу для стен
        let mut table = vec![vec![false; _width as usize]; _height as usize];

        for (i, node) in &self.nodes {
            let index = node.value as usize; // Предполагаем, что узлы нумеруются с 1
            let row = (index / width) * 2 + 1;
            let col = (index % width) * 2 + 1;

            // Добавляем узел
            table[row][col] = true;
            for &(neighbor_index, symbol) in &node.neighbors {
                /* N
                W     E
                S
                */
                if symbol == 'W' && (neighbor_index != index) {
                    // Вправо
                    table[row][col - 1] = true;
                } else if symbol == 'E' && (neighbor_index != index) {
                    // Влево
                    table[row][col + 1] = true;
                } else if symbol == 'S' && (neighbor_index != index) {
                    // Вниз
                    table[row + 1][col] = true;
                } else if symbol == 'N' && (neighbor_index != index) {
                    // Вверх
                    table[row - 1][col] = true;
                }
            }
        }

        // Добавляем стены
        for i in 0.._height {
            for j in 0.._width {
                if table[i as usize][j as usize] {
                    img.put_pixel(j, i, Rgba([255, 255, 255, 255])); // Белый цвет для узлов
                } else {
                    img.put_pixel(j, i, Rgba([0, 0, 0, 255])); // Черный цвет для стен
                }
            }
        }
        for (i, node) in &self.nodes {
            let index = node.value as usize; // Предполагаем, что узлы нумеруются с 1
            let row = (index / width) * 2 + 1;
            let col = (index % width) * 2 + 1;
            if (node.isFinal) {
                img.put_pixel(col as u32, row as u32, Rgba([0, 255, 0, 255]));
            }
        }
        // Добавляем зеленый квадрат в центре
        img.put_pixel(1, 1, Rgba([0, 255, 0, 255])); // Зеленый цвет

        // Сохраняем изображение в файл
        img.save(filename).expect("Failed to save image");
    }
    fn is_equivalent(&self, other: &Graph, s1: usize, s2: usize) -> bool {
        // Инициализация очереди для BFS
        let mut queue: Vec<(usize, usize)> = Vec::new();
        // Кладем в очередь начальные состояния
        queue.push((s1, s2)); // предполагаем, что начальные состояния имеют индекс 0
                              // Таблица использованных пар состояний
        let mut used: Vec<Vec<bool>> = vec![vec![false; other.nodes.len()]; self.nodes.len()];

        while !queue.is_empty() {
            // Извлекаем пару состояний из очереди
            let (u, v) = queue.remove(0);

            // Если состояния имеют разную терминальность, то автоматы не эквивалентны
            if self.nodes[&u].isFinal != other.nodes[&v].isFinal {
                return false;
            }
            used[u][v] = true;
            // Обходим соседние состояния для текущей пары
            for c in self.nodes[&u].neighbors.iter() {
                // Находим соответствующее состояние в другом автомате
                let neighbor_v = other.nodes[&v].neighbors.iter().find(|&&(_, nv)| nv == c.1);

                match neighbor_v {
                    Some((nv, _)) => {
                        // Если не использовали пару состояний (u', v'), то добавляем в очередь
                        if !used[c.0][*nv] {
                            queue.push((c.0, *nv));
                            used[c.0][*nv] = true;
                        }
                    }
                    None => {
                        // Если нашлось состояние, не имеющее соответствия в другом автомате,
                        // то автоматы не эквивалентны
                        return false;
                    }
                }
            }
        }

        // Если не найдено ни одной пары состояний, различающих автоматы, то они эквивалентны
        true
    }
    fn restore_path(
        &self,
        parent: HashMap<(usize, usize), Option<(usize, usize, char)>>,
        used: HashMap<(usize, usize), bool>,
        start: (usize, usize),
    ) -> Vec<char> {
        let mut path = Vec::new();
        let mut current = start;

        while let Some(true) = used.get(&current) {
            if let Some(option) = parent.get(&current) {
                if let Some((prev_u, prev_v, c)) = option.as_ref() {
                    path.push(*c);
                    current = (*prev_u, *prev_v);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        path.reverse();
        path
    }

    fn is_equivalent_counterexample(
        &self,
        other: &Graph,
        s1: usize,
        s2: usize,
        alphabet: Vec<char>,
    ) -> (bool, Vec<char>) {
        let mut queue: Vec<(usize, usize)> = Vec::new();
        queue.push((s1, s2));

        let mut used: HashMap<(usize, usize), bool> = HashMap::new();
        let mut parent: HashMap<(usize, usize), Option<(usize, usize, char)>> = HashMap::new();

        used.insert((s1, s2), true);

        while !queue.is_empty() {
            let (u, v) = queue.remove(0);

            if (self.nodes[&u].isFinal) {
                println!("{} {}", self.nodes[&u].isFinal, other.nodes[&v].isFinal);
            }
            if (other.nodes[&v].isFinal) {
                println!("{} {}", self.nodes[&u].isFinal, other.nodes[&v].isFinal);
            }
            if self.nodes[&u].isFinal != other.nodes[&v].isFinal {
                return (false, self.restore_path(parent, used, (u, v)));
            }

            for c in alphabet.iter() {
                let neighbor_u = self.nodes[&u].neighbors.iter().find(|&&(nu, nc)| nc == *c);
                let neighbor_v = other.nodes[&v].neighbors.iter().find(|&&(nv, nc)| nc == *c);

                match (neighbor_u, neighbor_v) {
                    (Some((nu, _)), Some((nv, _))) => {
                        if !used.contains_key(&(*nu, *nv)) {
                            queue.push((*nu, *nv));
                            used.insert((*nu, *nv), true);
                            parent.insert((*nu, *nv), Some((u, v, *c)));
                        }
                    }
                    (Some((nu, _)), None) => {
                        let mut ret_vec = self.restore_path(parent, used, (u, v));
                        ret_vec.push(*c);
                        ret_vec = self.dfs(*nu, &ret_vec, alphabet).expect("dfs warning ");
                        return (false, ret_vec);
                    }
                    (None, Some((nv, _))) => {
                        let mut ret_vec = self.restore_path(parent, used, (u, v));
                        ret_vec.push(*c);
                        ret_vec = other.dfs(*nv, &ret_vec, alphabet).expect("dfs warning ");
                        return (false, ret_vec);
                    }
                    (None, None) => {
                        continue;
                    }
                }
            }
        }

        (true, self.restore_path(parent, used, (s1, s2)))
    }

    fn from_table(
        main_prefixes: String,
        complementary_prefixes: String,
        suffixes: String,
        rows: Vec<bool>,
    ) {
    }

    pub fn is_equivalent_counterexample2(
        &self,
        other: &Graph,
        s1: usize,
        s2: usize,
        alphabet: Vec<char>,
    ) -> (bool, Vec<char>) {
        // Инициализация очереди для BFS
        let mut queue: Vec<(usize, usize, Vec<char>)> = Vec::new();
        // Кладем в очередь начальные состояния
        queue.push((s1, s2, Vec::new())); // предполагаем, что начальные состояния имеют индекс 0
                                          // Таблица использованных пар состояний
        let mut used: HashMap<(usize, usize), bool> = HashMap::new();
        let mut path = Vec::new();

        while !queue.is_empty() {
            // Извлекаем пару состояний из очереди
            let (u, v, mut parent_path) = queue.remove(0);

            // Если состояния имеют разную терминальность, то автоматы не эквивалентны
            if (self.nodes[&u].isFinal) {
                println!("{} {}", self.nodes[&u].isFinal, other.nodes[&v].isFinal);
            }
            if (other.nodes[&v].isFinal) {
                println!("{} {}", self.nodes[&u].isFinal, other.nodes[&v].isFinal);
            }
            if self.nodes[&u].isFinal != other.nodes[&v].isFinal {
                return (false, parent_path);
            }
            used.insert((u, v), true);
            // Обходим соседние состояния для текущей пары
            for c in alphabet.iter() {
                path = parent_path.clone();
                path.push(*c);
                // Находим соответствующее состояние в другом автомате
                let neighbor_u = self.nodes[&u].neighbors.iter().find(|&&(nu, nc)| nc == *c);
                let neighbor_v = other.nodes[&v].neighbors.iter().find(|&&(nv, nc)| nc == *c);

                match (neighbor_u, neighbor_v) {
                    (Some((nu, _)), Some((nv, _))) => {
                        // Если не использовали пару состояний (u', v'), то добавляем в очередь
                        if let Some(value) = used.get(&(*nu, *nv)) {
                            queue.push((*nu, *nv, path.clone()));

                            used.insert((*nu, *nv), true);
                        }
                    }
                    (Some(_), None) => {
                        // Если нашлось состояние, не имеющее соответствия в другом автомате,
                        // то автоматы не эквивалентны

                        return (false, path);
                    }
                    (None, Some(_)) => {
                        // Если нашлось состояние, не имеющее соответствия в другом автомате,
                        // то автоматы не эквивалентны
                        return (false, path);
                    }
                    (None, None) => {
                        // Если для обоих автоматов нет перехода по символу c, то все нормально
                        continue;
                    }
                }
            }
        }

        // Если не найдено ни одной пары состояний, различающих автоматы, то они эквивалентны
        (true, path)
    }

    fn dfs(&self, state: usize, path: &Vec<char>, alphabet: Vec<char>) -> Option<Vec<char>> {
        if self.nodes[&state].isFinal {
            return Some(path.to_vec());
        }

        let mut visited: HashSet<usize> = HashSet::new();
        let mut stack: Vec<(usize, Vec<char>)> = vec![(state, path.to_vec())];

        while let Some((current_state, current_path)) = stack.pop() {
            if visited.contains(&current_state) {
                continue;
            }
            visited.insert(current_state);

            for c in alphabet.iter() {
                let neighbor = self.nodes[&current_state]
                    .neighbors
                    .iter()
                    .find(|&&(neighbor_state, neighbor_char)| neighbor_char == *c);
                if let Some((neighbor_state, _)) = neighbor {
                    let mut new_path = current_path.clone();
                    new_path.push(*c);

                    if self.nodes[&neighbor_state].isFinal {
                        return Some(new_path);
                    }
                    stack.push((*neighbor_state, new_path));
                }
            }
        }

        None
    }
}

enum Side {
    North,
    South,
    East,
    West,
}

impl Side {
    fn abbreviation(&self) -> char {
        match self {
            Side::North => 'N',
            Side::South => 'S',
            Side::East => 'E',
            Side::West => 'W',
        }
    }

    fn full_name(&self) -> &'static str {
        match self {
            Side::North => "North",
            Side::South => "South",
            Side::East => "East",
            Side::West => "West",
        }
    }

    fn opposite(&self) -> Side {
        match self {
            Side::North => Side::South,
            Side::South => Side::North,
            Side::East => Side::West,
            Side::West => Side::East,
        }
    }
}
fn rand_dfs_rect(
    graph: &mut Graph,
    start_node_index: usize,
    pr_of_break_wall: usize,
    width: usize,
    height: usize,
) {
    let mut stack = Vec::new();
    let mut visited = vec![false; (width + 2) * (height + 2)]; // Массив для отслеживания посещённых узлов
    stack.push(start_node_index);

    while !stack.is_empty() {
        // Получаем текущий узел из стека
        let node_index = *stack.last().unwrap();

        visited[node_index] = true; // Отмечаем узел как посещённый

        // Получаем соседей текущего узла
        let mut neighbors = Vec::new();

        let north = node_index as i32 - (width + 2) as i32;
        let south = node_index as i32 + (width + 2) as i32;
        let east = node_index as i32 + 1;
        let west = node_index as i32 - 1;

        // Проверяем соседей и добавляем их в список
        if south > (width + 2) as i32
            && south < (width as i32 + 2) * (height as i32 + 1)
            && !visited[south as usize]
        {
            neighbors.push((south as usize, Side::South));
        }
        if north > (width + 2) as i32
            && north < (width as i32 + 2) * (height as i32 + 1)
            && !visited[north as usize]
        {
            neighbors.push((north as usize, Side::North));
        }
        if (east % (width + 2) as i32) > 0
            && (east % (width + 2) as i32) < (width + 1) as i32
            && !visited[east as usize]
        {
            neighbors.push((east as usize, Side::East));
        }
        if (west % (width + 2) as i32) > 0 && !visited[west as usize] {
            neighbors.push((west as usize, Side::West));
        }

        if neighbors.is_empty() {
            let mut _neighbors = Vec::new();
            let mut neighbors_iter = graph.nodes[&node_index].neighbors.iter();

            if north > (width + 2) as i32
                && north < (width as i32 + 2) * (height as i32 + 1)
                && !neighbors_iter.any(|&x| x.0 == north as usize)
            {
                _neighbors.push((north as usize, Side::North));
            }
            if south > (width + 2) as i32
                && south < (width as i32 + 2) * (height as i32 + 1)
                && !neighbors_iter.any(|&x| x.0 == south as usize)
            {
                _neighbors.push((south as usize, Side::South));
            }
            if (east % (width + 2) as i32) > 0
                && (east % (width + 2) as i32) < (width + 1) as i32
                && !neighbors_iter.any(|&x| x.0 == east as usize)
            {
                _neighbors.push((east as usize, Side::East));
            }
            if (west % (width + 2) as i32) > 1 && !neighbors_iter.any(|&x| x.0 == west as usize) {
                _neighbors.push((west as usize, Side::West));
            }
            let p = generate_random_number(100);
            if (_neighbors.len() > 0) {
                if p < pr_of_break_wall as u32 {
                    let i = generate_random_number(_neighbors.len() as u32);

                    // Добавляем ребра
                    graph.add_edge(
                        node_index,
                        _neighbors[i as usize].0,
                        _neighbors[i as usize].1.abbreviation(),
                    );
                    graph.add_edge(
                        _neighbors[i as usize].0,
                        node_index,
                        _neighbors[i as usize].1.opposite().abbreviation(),
                    );
                }
            }
            stack.pop(); // Если нет соседей, удаляем текущий узел из стека
            continue; // Переходим к следующему узлу в стеке
        }

        // Перемешиваем соседей для случайного выбора
        let i = generate_random_number(neighbors.len() as u32);

        // Добавляем ребра
        graph.add_edge(
            node_index,
            neighbors[i as usize].0,
            neighbors[i as usize].1.abbreviation(),
        );
        graph.add_edge(
            neighbors[i as usize].0,
            node_index,
            neighbors[i as usize].1.opposite().abbreviation(),
        );

        // Добавляем выбранного соседа в стек
        stack.push(neighbors[i as usize].0);
    }
    let hw = (height + 1) * (width + 2);
    for i in 0..=width {
        graph.add_edge(i, (i + 1), 'E');
        graph.add_edge((i + 1), i, 'W');

        graph.add_edge(i + hw, (i + 1) + hw, 'E');
        graph.add_edge((i + 1) + hw, i + hw, 'W');
    }

    for i in 0..=height {
        // Левая и правая границы
        graph.add_edge(i * (width + 2), (i + 1) * (width + 2), 'S');
        graph.add_edge((i + 1) * (width + 2), i * (width + 2), 'N');

        graph.add_edge(
            i * (width + 2) + width + 1,
            (i + 1) * (width + 2) + width + 1,
            'S',
        );
        graph.add_edge(
            (i + 1) * (width + 2) + width + 1,
            i * (width + 2) + width + 1,
            'N',
        );
    }
}
static mut GLOBAL_RNG: Option<rand::prelude::ThreadRng> = None;
fn generate_random_number(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }

    unsafe { GLOBAL_RNG.as_mut().unwrap().gen_range(0..=n - 1) }
}
pub fn fill_rand_dfs_rect(
    graph: &mut Graph,
    num_of_finish_edge: usize,
    pr_of_break_wall: usize,
    width: usize,
    height: usize,
) {
    for i in 0..height + 2 {
        for j in 0..width + 2 {
            let value = i * (width + 2) + j;
            graph.add_node(
                value,
                (i == 0 || i == height + 1 || j == 0 || j == width + 1),
            );
        }
    }
    rand_dfs_rect(graph, width + 3, pr_of_break_wall, width, height);

    // Генерация выходных рёбер
    let hw = (height + 1) * (width + 2);
    let mut exits_added = 0;

    while exits_added < num_of_finish_edge {
        let side = generate_random_number(4); // случайно выбираем сторону
        let position: usize = match side {
            0 => generate_random_number(width as u32 + 1) as usize, // Верхняя граница
            1 => (height + 1) * (width + 2) + generate_random_number(width as u32 + 1) as usize, // Нижняя граница
            2 => (generate_random_number(height as u32 + 1) as usize) * (width + 2), // Левая граница
            _ => (generate_random_number(height as u32 + 1) as usize) * (width + 2) + width + 1, // Правая граница
        };

        // Проверка, что на данном месте ещё нет выхода
        let opposite_position = match side {
            0 => position + (width + 2), // Выход вниз
            1 => position - (width + 2), // Выход вверх
            2 => position + 1,           // Выход вправо
            _ => position - 1,           // Выход влево
        };

        // Проверяем, есть ли ребро между текущей позицией и противоположной
        if !graph.has_edge(position, opposite_position) {
            // Добавляем выход
            match side {
                0 => {
                    graph.add_edge(position, opposite_position, 'S'); // Добавляем ребро на выход вниз
                    graph.add_edge(opposite_position, position, 'N'); // Добавляем ребро на выход вниз
                }
                1 => {
                    graph.add_edge(position, opposite_position, 'N');
                    graph.add_edge(opposite_position, position, 'S');
                } // Добавляем ребро на выход вверх
                2 => {
                    graph.add_edge(position, opposite_position, 'E');
                    graph.add_edge(opposite_position, position, 'W');
                } // Добавляем ребро на выход вправо
                _ => {
                    graph.add_edge(position, opposite_position, 'W');
                    graph.add_edge(opposite_position, position, 'E');
                } // Добавляем ребро на выход влево
            }
            exits_added += 1; // Увеличиваем счётчик добавленных выходов
        }
    }
}
/*
fn main2() {
    let mut graph = Graph::new();
    let pr_of_break_wall = 10;
    let pr_of_blocked_node = 15;
    let num_of_finish_edge = 10;
    let width = 100;
    let height = 100;

    unsafe {
        GLOBAL_RNG = Some(rand::thread_rng());
    }
    fill_rand_dfs_rect(
        &mut graph,
        num_of_finish_edge,
        pr_of_break_wall,
        width,
        height,
    );

    // Create a Hilbert curve of level 3 in a grid of size 8x8
    graph.print_png_rect(width + 2, height + 2, "test.png");
    graph.print_json_to_file("graph1.json");
}

fn test0() -> Result<(), serde_json::Error> {
    let json_string = fs::read_to_string("graph1.json").expect("sad");

    let mut graph: Graph = Graph::from_json_string(&json_string.to_string())?;
    //graph.print_dot();
    graph.print_json_to_file("graph2.json");
    graph.print_png_rect(102, 102, "graph12.png");
    //graph.print_dot();
    Ok(())
}

fn test1() {
    // Создаем первый граф
    let mut graph1 = Graph::new();
    graph1.add_node(0, false);
    graph1.add_node(1, false);
    graph1.add_node(2, false);
    graph1.add_node(3, false);
    graph1.add_node(4, false);
    graph1.add_node(5, true);

    graph1.add_edge(0, 1, 'a');
    graph1.add_edge(1, 2, 'b');
    graph1.add_edge(1, 3, 'a');
    graph1.add_edge(3, 4, 'b');
    graph1.add_edge(2, 4, 'b');
    graph1.add_edge(4, 5, 'a');

    // Создаем второй граф
    let mut graph2 = Graph::new();
    graph2.add_node(0, false);
    graph2.add_node(1, false);
    graph2.add_node(2, false);
    graph2.add_node(3, false);
    graph2.add_node(4, true);

    graph2.add_edge(0, 1, 'a');
    graph2.add_edge(1, 2, 'b');
    graph2.add_edge(1, 2, 'a');
    graph2.add_edge(2, 3, 'b');
    graph2.add_edge(3, 4, 'a');

    // Проверяем эквивалентность графиков
    graph1.print_dot();
    graph2.print_dot();
    println!("{}", graph1.is_equivalent(&graph2, 0, 0));
}

fn test2() {
    // Создаем первый граф
    let mut graph1 = Graph::new();
    graph1.add_node(0, false);
    graph1.add_node(1, false);
    graph1.add_node(2, false);
    graph1.add_node(3, true);

    graph1.add_edge(0, 1, 'a');
    graph1.add_edge(1, 2, 'a');
    graph1.add_edge(2, 3, 'b');

    // Создаем второй граф
    let mut graph2 = Graph::new();
    graph2.add_node(0, false);
    graph2.add_node(1, true);

    graph2.add_edge(0, 0, 'a');
    graph2.add_edge(0, 1, 'b');

    let alfabet = vec!['a', 'b'];
    // Проверяем эквивалентность графиков
    graph1.print_dot();
    graph2.print_dot();
    let (isEq, vec_c_examp) = graph1.is_equivalent_counterexample2(&graph2, 0, 0, alfabet);
    println!("{}", isEq);
    if (!isEq) {
        for (i, symbol) in vec_c_examp.iter().enumerate() {
            print!("{}", symbol);
        }
        // Найдем тот символ, который создает контр-пример
    }
}

fn test3() {
    let mut graph = Graph::new();
    let pr_of_break_wall = 10;
    let num_of_finish_edge = 5;
    let width = 10;
    let height = 10;

    unsafe {
        GLOBAL_RNG = Some(rand::thread_rng());
    }
    fill_rand_dfs_rect(
        &mut graph,
        num_of_finish_edge,
        pr_of_break_wall,
        width,
        height,
    );

    // Create a Hilbert curve of level 3 in a grid of size 8x8
    graph.print_json_to_file("graph_test.json");
    graph.print_png_rect(width + 2, height + 2, "graph_test1.png");

    let json_string = fs::read_to_string("graph1.json").expect("sad");

    let mut graph1: Graph =
        Graph::from_json_string(&json_string.to_string()).expect("wrong graph json ");
    //graph.print_dot();
    graph1.print_json_to_file("graph2.json");
    graph1.print_png_rect(102, 102, "graph_test2.png");

    let alphabet = vec!['N', 'S', 'W', 'E'];
    graph1.complete_with_alphabet(alphabet.clone());
    graph.complete_with_alphabet(alphabet.clone());
    let (isEq, vec_c_examp) = graph.is_equivalent_counterexample(&graph1, width + 3, 103, alphabet);
    println!("{}", isEq);

    let mut word = String::new();
    for symbol in vec_c_examp.iter() {
        word.push(*symbol);
    }
    println!("{}", word);
    println!("{}", graph.travers(width + 3, word.clone()));
    println!("{}", graph1.travers(103, word.clone()));
    // Найдем тот символ, который создает контр-пример
}

fn main() {
    let width = 10;
    let height = 10;
    // Create a Hilbert curve of level 3 in a grid of size 8x8
    let json_string = fs::read_to_string("graph_test.json").expect("sad");
    let mut graph = Graph::from_json_string(&json_string.to_string()).expect("wrong graph json ");

    let json_string = fs::read_to_string("graph1.json").expect("sad");

    let mut graph1: Graph =
        Graph::from_json_string(&json_string.to_string()).expect("wrong graph json ");
    //graph.print_dot();
    graph1.print_json_to_file("graph2.json");

    let alphabet = vec!['N', 'S', 'W', 'E'];
    graph1.complete_with_alphabet(alphabet.clone());
    graph.print_png_rect(width + 2, height + 2, "graph_test1.png");

    graph1.print_png_rect(102, 102, "graph_test2.png");
    graph.complete_with_alphabet(alphabet.clone());
    let (isEq, vec_c_examp) = graph.is_equivalent_counterexample(&graph1, width + 3, 103, alphabet);
    println!("{}", isEq);

    let mut word = String::new();
    for symbol in vec_c_examp.iter() {
        word.push(*symbol);
    }
    println!("{}", word);
    println!("{}", graph.travers(width + 3, word.clone()));
    println!("{}", graph1.travers(103, word.clone()));
    // Найдем тот символ, который создает контр-пример
}
*/
use std::sync::{Arc, Mutex};

static mut GRAPH: Option<Arc<Mutex<Graph>>> = None;
static mut WIDTH: Option<Arc<Mutex<usize>>> = None;
static mut HEIGHT: Option<Arc<Mutex<usize>>> = None;

fn init_graph() {
    unsafe {
        GRAPH = Some(Arc::new(Mutex::new(Graph::new())));
        WIDTH = Some(Arc::new(Mutex::new(0)));
        HEIGHT = Some(Arc::new(Mutex::new(0)));
    }
}

// cпасибо вове за кусочек кода
pub fn dfa_from_lstar(
    main_prefixes: &[String],
    complementary_prefixes: &[String],
    suffixes: &[String],
    rows: Vec<String>,
) -> (usize, Graph) {
    let mut graph = Graph::new();
    let mut state_map = HashMap::new();
    let mut prefix_to_row = HashMap::new();
    let mut row_to_prefix = HashMap::new();
    let mut q0 = None;

    for (idx, p) in main_prefixes.iter().enumerate() {
        state_map.insert(p.to_string(), idx);
        let is_final = rows[idx].chars().next().unwrap().to_digit(10).unwrap() == 1;

        graph.add_node(idx, is_final);
        if (p.to_string() == "".to_string()) {
            if q0.is_none() {
                q0 = Some(idx);
            }
        }
    }

    for (idx, p) in main_prefixes.iter().enumerate() {
        prefix_to_row.insert(p.to_string(), rows[idx].as_str());
        row_to_prefix.insert(rows[idx].as_str(), p.to_string());
    }

    for (idx, p) in complementary_prefixes.iter().enumerate() {
        prefix_to_row.insert(p.to_string(), rows[main_prefixes.len() + idx].as_str());
    }

    for p in main_prefixes.iter().chain(complementary_prefixes) {
        if p.len() > 0 {
            let sub_prefix = &p[0..p.len() - 1];
            let letter = p.chars().last().unwrap();
            let from = *state_map.get(sub_prefix).unwrap_or(&0);
            let to = *state_map.get(p).unwrap_or(&0);

            graph.add_edge(from, to, letter);
        }
    }
    graph.print_dot();
    graph.print_json_to_file("table.json");
    (q0.expect("smt with table"), graph)
}

#[post("/generate_graph")]
pub async fn generate_graph(data: Json<GenerateGraphRequest>) -> HttpResponse {
    init_graph(); // инициализируем граф, если он еще не инициализирован

    let mut graph = Graph::new();

    unsafe {
        GLOBAL_RNG = Some(rand::thread_rng());
    }
    fill_rand_dfs_rect(
        &mut graph,
        data.num_of_finish_edge,
        data.pr_of_break_wall,
        data.width,
        data.height,
    );
    println!(
        "{} {} {} {}",
        data.num_of_finish_edge, data.pr_of_break_wall, data.width, data.height
    );

    let alphabet = vec!['N', 'S', 'W', 'E'];
    graph.complete_with_alphabet(alphabet.clone());

    let graph_string = graph.to_json_string();
    unsafe {
        *WIDTH.as_ref().unwrap().lock().unwrap() = data.width;
        *HEIGHT.as_ref().unwrap().lock().unwrap() = data.height;
        *GRAPH.as_ref().unwrap().lock().unwrap() = graph
    };

    HttpResponse::Ok()
        .header(header::CONTENT_TYPE, "application/json")
        .body(graph_string)
}

#[get("/get_graph")]
pub async fn get_graph() -> HttpResponse {
    // инициализируем граф, если он еще не инициализирован
    let mut graph = unsafe { GRAPH.as_ref().unwrap().lock().unwrap() };
    let mut _width = unsafe { WIDTH.as_ref().unwrap().lock().unwrap() };
    let mut _height = unsafe { HEIGHT.as_ref().unwrap().lock().unwrap() };
    let graph_string = graph.to_json_string();
    graph.print_png_rect(*_width + 2, *_height + 2, "get_graph.png");
    graph.print_json_to_file("get_graph.json");
    HttpResponse::Ok()
        .header(header::CONTENT_TYPE, "application/json")
        .body(graph_string)
}

#[derive(Deserialize)]
struct CheckAutomataRequest {
    width: usize,
    height: usize,
    startpoint: usize,
    graph_data: GraphData,
}

#[post("/check_automata")]
pub async fn check_automata(data: Json<CheckAutomataRequest>) -> HttpResponse {
    // инициализируем граф, если он еще не инициализирован
    let mut graph1: Graph = Graph::from_json(&data.graph_data).expect("wrong graph json ");
    let mut graph = unsafe { GRAPH.as_ref().unwrap().lock().unwrap() };
    let mut _width = unsafe { WIDTH.as_ref().unwrap().lock().unwrap() };
    let mut _height = unsafe { HEIGHT.as_ref().unwrap().lock().unwrap() };
    let alphabet = vec!['N', 'S', 'W', 'E'];
    // graph.print_png_rect(*_width +2, *_height+2, "post_genrated");
    // graph1.print_png_rect(data.width +2, data.height+2, "post_genrated");
    let (isEq, vec_c_examp) =
        graph.is_equivalent_counterexample2(&graph1, *_width + 3, data.startpoint, alphabet);
    if isEq {
        HttpResponse::Ok().json("true")
    } else {
        let mut word = String::new();
        for symbol in vec_c_examp.iter() {
            word.push(*symbol);
        }
        HttpResponse::Ok().json(word)
    }
}

#[derive(Deserialize)]
struct CheckTableRequest {
    main_prefixes: String,
    complementary_prefixes: String,
    suffixes: String,
    table: String,
}

#[post("/check_table")]
pub async fn check_table(data: Json<CheckTableRequest>) -> HttpResponse {
    // инициализируем граф, если он еще не инициализирован
    // разделение строк
    let main_prefix_strings: Vec<String> = data
        .main_prefixes
        .split_ascii_whitespace()
        .map(|s| s.replace("e", ""))
        .map(String::from)
        .collect();
    let complementary_prefix_strings: Vec<String> = data
        .complementary_prefixes
        .split_ascii_whitespace()
        .map(|s| s.replace("e", ""))
        .map(String::from)
        .collect();
    let suffix_strings: Vec<String> = data
        .suffixes
        .split_ascii_whitespace()
        .map(|s| s.replace("e", ""))
        .map(String::from)
        .collect();
    let table_rows: Vec<String> = data
        .table
        .chars()
        .collect::<Vec<char>>()
        .chunks(suffix_strings.len())
        .map(|chunk| chunk.iter().collect::<String>())
        .collect();

    // инициализация графов
    let (mut startpoint, mut graph1) = dfa_from_lstar(
        &main_prefix_strings,
        &complementary_prefix_strings,
        &suffix_strings,
        table_rows,
    );
    let mut graph = unsafe { GRAPH.as_ref().unwrap().lock().unwrap() };
    let mut _width = unsafe { WIDTH.as_ref().unwrap().lock().unwrap() };
    let mut _height = unsafe { HEIGHT.as_ref().unwrap().lock().unwrap() };
    let alphabet = vec!['N', 'S', 'W', 'E'];
    // graph.print_png_rect(*_width +2, *_height+2, "post_genrated");
    // graph1.print_png_rect(data.width +2, data.height+2, "post_genrated");
    let (isEq, vec_c_examp) =
        graph.is_equivalent_counterexample(&graph1, *_width + 3, startpoint, alphabet);
    if isEq {
        HttpResponse::Ok()
            .header(header::CONTENT_TYPE, "application/json")
            .body("true")
    } else {
        let mut word = String::new();
        for symbol in vec_c_examp.iter() {
            word.push(*symbol);
        }
        HttpResponse::Ok()
            .header(header::CONTENT_TYPE, "application/json")
            .body(word)
    }
}

#[post("/check_membership")]
pub async fn check_membership(path: String) -> HttpResponse {
    let mut graph = unsafe { GRAPH.as_ref().unwrap().lock().unwrap() };
    let mut _width = unsafe { WIDTH.as_ref().unwrap().lock().unwrap() };
    let mut _height = unsafe { HEIGHT.as_ref().unwrap().lock().unwrap() };

    //let path: Vec<char> = _path.chars().collect();
    let result = graph.travers(*_width + 3, path);
    if result {
        HttpResponse::Ok()
            .header(header::CONTENT_TYPE, "application/json")
            .body("1")
    } else {
        HttpResponse::Ok()
            .header(header::CONTENT_TYPE, "application/json")
            .body("0")
    }
}
#[get("/get_path")]
pub async fn get_path() -> HttpResponse {
    let mut graph = unsafe { GRAPH.as_ref().unwrap().lock().unwrap() };
    let mut _width = unsafe { WIDTH.as_ref().unwrap().lock().unwrap() };
    let mut _height = unsafe { HEIGHT.as_ref().unwrap().lock().unwrap() };
    let alphabet = vec!['N', 'S', 'W', 'E'];
    let startpoint = *_width + 3;
    let path = graph.dfs(startpoint, &Vec::new(), alphabet.clone());
    match path {
        Some(path) => HttpResponse::Ok()
            .header(header::CONTENT_TYPE, "application/json")
            .body(path.iter().collect::<String>()),
        None => HttpResponse::Ok()
            .header(header::CONTENT_TYPE, "application/json")
            .body(""),
    }
}
