package planner

import (
	"fmt"

	"github.com/xwb1989/sqlparser"
	"mit.edu/dsg/godb/catalog"
)

// SQLPlanner orchestrates the parsing, logical planning, optimization,
// and physical planning steps.
type SQLPlanner struct {
	catalog       *catalog.Catalog
	opt           *Optimizer
	physicalRules []PhysicalConversionRule
}

func NewSQLPlanner(c *catalog.Catalog, logicalRules []LogicalRule, physicalRules []PhysicalConversionRule) *SQLPlanner {
	return &SQLPlanner{
		catalog:       c,
		opt:           NewOptimizer(logicalRules),
		physicalRules: physicalRules,
	}
}

// Plan takes a SQL string and returns a constructed Physical Plan tree.
// It handles the pipeline: SQL -> AST -> Logical -> Optimized -> Physical.
func (p *SQLPlanner) Plan(sql string, silent bool) (PlanNode, error) {
	stmt, err := sqlparser.Parse(sql)
	if err != nil {
		return nil, fmt.Errorf("parse error: %w", err)
	}

	lb := NewLogicalPlanBuilder(p.catalog)
	logicalPlan, err := lb.Plan(stmt)
	if err != nil {
		return nil, fmt.Errorf("logical planning error: %w", err)
	}

	if !silent {
		fmt.Printf("Initial Logical Plan:\n%s\n", PrettyPrintLogicalPlan(logicalPlan))
	}

	optimizedPlan := p.opt.Optimize(logicalPlan)

	if !silent {
		fmt.Printf("Optimized Logical Plan:\n%s\n", PrettyPrintLogicalPlan(optimizedPlan))
	}

	pb := NewPhysicalPlanBuilder(p.catalog, p.physicalRules)
	physicalPlan, err := pb.Build(optimizedPlan)
	if err != nil {
		return nil, fmt.Errorf("physical planning error: %w", err)
	}

	return physicalPlan, nil
}
